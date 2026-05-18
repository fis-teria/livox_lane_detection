#include <torch/script.h>
#if defined(LIVOX_LANE_DETECTION_TORCH_CUDA)
#include <torch/cuda.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"

namespace
{

using SteadyClock = std::chrono::steady_clock;

double MillisecondsBetween(
  const SteadyClock::time_point & start,
  const SteadyClock::time_point & end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string ToLowerAscii(std::string value)
{
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  return value;
}

torch::Device ResolveTorchDevice(const std::string & requested_device, const rclcpp::Logger & logger)
{
  const auto requested = ToLowerAscii(requested_device);
  const auto cuda_requested = requested == "auto" || requested == "cuda" ||
    requested.rfind("cuda:", 0) == 0;

  if (!cuda_requested) {
    return torch::Device(torch::kCPU);
  }

  bool cuda_available = false;
#if defined(LIVOX_LANE_DETECTION_TORCH_CUDA)
  try {
    cuda_available = torch::cuda::is_available();
  } catch (const c10::Error & error) {
    RCLCPP_WARN(
      logger, "CUDA availability check failed, falling back to CPU: %s", error.what());
  }
#else
  if (requested != "auto") {
    RCLCPP_WARN(
      logger,
      "Requested livox_lane_detection device '%s' but this node was built with CPU-only "
      "libtorch. Falling back to CPU.",
      requested_device.c_str());
  }
#endif

  if (!cuda_available) {
#if defined(LIVOX_LANE_DETECTION_TORCH_CUDA)
    if (requested != "auto") {
      RCLCPP_WARN(
        logger,
        "Requested livox_lane_detection device '%s' but CUDA is not available in this "
        "libtorch/runtime. Falling back to CPU.",
        requested_device.c_str());
    }
#endif
    return torch::Device(torch::kCPU);
  }

  int device_index = 0;
  if (requested.rfind("cuda:", 0) == 0) {
    try {
      device_index = std::stoi(requested.substr(5));
    } catch (const std::exception & error) {
      RCLCPP_WARN(
        logger, "Invalid CUDA device '%s' (%s), using cuda:0.",
        requested_device.c_str(), error.what());
      device_index = 0;
    }
  }

  return torch::Device(torch::kCUDA, device_index);
}

struct PointXYZI
{
  float x {};
  float y {};
  float z {};
  float intensity {};
};

struct BvCommonSettings
{
  float train_height_shift {0.0F};
  float shifted_min_height {-1.0F};
  float shifted_max_height {1.0F};
  float distance_resolution_train {6.0F};
  float width_resolution_train {30.0F};
  float point_radius_train {1.5F};
  float truncation_max_intensity {0.12F};
  float train_background_intensity_shift {0.1F};
};

struct BvRangeSettings
{
  float max_distance {100.0F};
  float min_distance {-40.0F};
  float left_distance {20.0F};
  float right_distance {20.0F};
};

struct BvTensor
{
  int height {0};
  int width {0};
  std::vector<float> data {};
};

constexpr std::array<std::array<std::uint8_t, 3>, 33> kClassColors {{
  {10, 10, 10},
  {0, 255, 233},
  {0, 161, 255},
  {106, 255, 0},
  {0, 173, 26},
  {193, 98, 94},
  {121, 138, 34},
  {0, 38, 232},
  {234, 66, 241},
  {56, 140, 0},
  {173, 6, 95},
  {0, 60, 255},
  {0, 125, 200},
  {100, 20, 255},
  {50, 60, 150},
  {98, 146, 99},
  {255, 201, 29},
  {191, 135, 43},
  {177, 89, 40},
  {240, 112, 71},
  {57, 132, 169},
  {200, 100, 60},
  {67, 142, 169},
  {116, 206, 117},
  {117, 89, 40},
  {189, 251, 107},
  {142, 59, 153},
  {36, 28, 237},
  {111, 99, 83},
  {255, 104, 152},
  {0, 128, 255},
  {0, 64, 128},
  {124, 95, 13},
}};

BvRangeSettings GetBvRangeSettings(const std::vector<std::string> & lidar_ids)
{
  BvRangeSettings settings;
  const auto has = [&lidar_ids](const std::string & id) {
      return std::find(lidar_ids.begin(), lidar_ids.end(), id) != lidar_ids.end();
    };

  if (has("6")) {
    settings.max_distance = 100.0F;
  } else if (has("1") || has("2") || has("5")) {
    settings.max_distance = 40.0F;
  } else {
    settings.max_distance = 0.0F;
  }

  if (!(has("2") || has("3") || has("1") || has("6"))) {
    settings.left_distance = 0.0F;
  }
  if (!(has("4") || has("5") || has("1") || has("6"))) {
    settings.right_distance = 0.0F;
  }
  if (!(has("3") || has("4"))) {
    settings.min_distance = 0.0F;
  }

  return settings;
}

BvTensor ProduceBvData(
  const std::vector<PointXYZI> & points, const BvCommonSettings & common,
  const BvRangeSettings & range)
{
  const int bv_width = static_cast<int>(
    (range.right_distance + range.left_distance) * common.width_resolution_train);
  const int bv_height = static_cast<int>(
    (range.max_distance - range.min_distance) * common.distance_resolution_train);

  BvTensor tensor;
  tensor.height = bv_height;
  tensor.width = bv_width;
  tensor.data.assign(static_cast<std::size_t>(2 * bv_width * bv_height), 0.0F);

  const float max_distance_px = range.max_distance * common.distance_resolution_train;
  const float left_distance_px = range.left_distance * common.width_resolution_train;
  const int radius = static_cast<int>(std::ceil(common.point_radius_train));

  for (const auto & point : points) {
    const float shifted_z = point.z + common.train_height_shift;
    if (shifted_z < common.shifted_min_height || shifted_z > common.shifted_max_height) {
      continue;
    }

    float normalized_intensity = point.intensity / 255.0F;
    normalized_intensity = std::min(normalized_intensity, common.truncation_max_intensity);
    normalized_intensity /= common.truncation_max_intensity;
    normalized_intensity += common.train_background_intensity_shift;

    const int image_x = static_cast<int>(
      -point.y * common.width_resolution_train + left_distance_px);
    const int image_y = static_cast<int>(
      max_distance_px - point.x * common.distance_resolution_train);

    if (image_x < 0 || image_x >= bv_width || image_y < 0 || image_y >= bv_height) {
      continue;
    }

    const int start_x = std::max(0, image_x - radius);
    const int end_x = std::min(bv_width - 1, image_x + radius);
    const int start_y = std::max(0, image_y - radius);
    const int end_y = std::min(bv_height - 1, image_y + radius);

    for (int y = start_y; y <= end_y; ++y) {
      for (int x = start_x; x <= end_x; ++x) {
        const std::size_t index = static_cast<std::size_t>(y * bv_width + x);
        tensor.data[index] = std::max(tensor.data[index], normalized_intensity);
        tensor.data[static_cast<std::size_t>(bv_width * bv_height) + index] =
          std::max(
          tensor.data[static_cast<std::size_t>(bv_width * bv_height) + index],
          shifted_z);
      }
    }
  }

  return tensor;
}

torch::Tensor ForwardModule(torch::jit::script::Module & module, torch::Tensor input)
{
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(std::move(input));
  auto output = module.forward(inputs);

  if (output.isTensor()) {
    return output.toTensor();
  }
  if (output.isTuple()) {
    const auto & elements = output.toTuple()->elements();
    if (elements.empty() || !elements.back().isTensor()) {
      throw std::runtime_error("TorchScript module returned an unexpected tuple.");
    }
    return elements.back().toTensor();
  }

  throw std::runtime_error("TorchScript module returned an unsupported type.");
}

std::vector<std::int64_t> RunInference(
  torch::jit::script::Module & module,
  const BvTensor & bv_data,
  const torch::Device & device)
{
  torch::NoGradGuard no_grad;

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto input = torch::from_blob(
    const_cast<float *>(bv_data.data.data()),
    {1, 2, bv_data.height, bv_data.width},
    options).clone();
  if (device.type() != torch::kCPU) {
    input = input.to(device);
  }

  auto output = ForwardModule(module, input);
  auto label_map = output.argmax(1).squeeze().to(torch::kCPU).contiguous();

  const auto numel = label_map.numel();
  std::vector<std::int64_t> result(static_cast<std::size_t>(numel));
  std::memcpy(
    result.data(), label_map.data_ptr<std::int64_t>(),
    sizeof(std::int64_t) * static_cast<std::size_t>(numel));
  return result;
}

std::vector<float> GetPointsClassFromBv(
  const std::vector<PointXYZI> & points, const std::vector<std::int64_t> & bv_label_map,
  const BvCommonSettings & common, const BvRangeSettings & range, int bv_width, int bv_height)
{
  std::vector<float> point_classes(points.size(), 0.0F);
  const float max_distance_px = range.max_distance * common.distance_resolution_train;
  const float left_distance_px = range.left_distance * common.width_resolution_train;

  for (std::size_t i = 0; i < points.size(); ++i) {
    const float shifted_z = points[i].z + common.train_height_shift;
    if (shifted_z < common.shifted_min_height || shifted_z > common.shifted_max_height) {
      continue;
    }

    const int image_x = static_cast<int>(
      -points[i].y * common.width_resolution_train + left_distance_px);
    const int image_y = static_cast<int>(
      max_distance_px - points[i].x * common.distance_resolution_train);

    if (image_x >= 0 && image_x < bv_width && image_y >= 0 && image_y < bv_height) {
      point_classes[i] = static_cast<float>(
        bv_label_map[static_cast<std::size_t>(image_y * bv_width + image_x)]);
    }
  }

  return point_classes;
}

std::uint32_t PackRgb(const std::array<std::uint8_t, 3> & color)
{
  return (static_cast<std::uint32_t>(color[0]) << 16) |
         (static_cast<std::uint32_t>(color[1]) << 8) |
         static_cast<std::uint32_t>(color[2]);
}

std::vector<PointXYZI> ReadPointsFromCloud(const sensor_msgs::msg::PointCloud2 & cloud)
{
  std::vector<PointXYZI> points;
  if (cloud.width == 0U || cloud.height == 0U) {
    return points;
  }

  points.reserve(static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height));

  const bool has_intensity = std::any_of(
    cloud.fields.begin(), cloud.fields.end(),
    [](const sensor_msgs::msg::PointField & field) {return field.name == "intensity";});

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

  if (has_intensity) {
    sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(cloud, "intensity");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
      if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
        points.push_back(PointXYZI {*iter_x, *iter_y, *iter_z, *iter_intensity});
      }
    }
  } else {
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
        points.push_back(PointXYZI {*iter_x, *iter_y, *iter_z, 0.0F});
      }
    }
  }

  return points;
}

sensor_msgs::msg::PointCloud2 BuildColoredCloud(
  const std::vector<PointXYZI> & points, const std::vector<float> & classes,
  const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::PointCloud2 output;
  output.header = header;
  output.height = 1U;
  output.width = static_cast<std::uint32_t>(points.size());
  output.is_dense = false;
  output.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier modifier(output);
  modifier.setPointCloud2Fields(
    6,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
    "rgb", 1, sensor_msgs::msg::PointField::UINT32,
    "class", 1, sensor_msgs::msg::PointField::FLOAT32);
  modifier.resize(points.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(output, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(output, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(output, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_intensity(output, "intensity");
  sensor_msgs::PointCloud2Iterator<std::uint32_t> iter_rgb(output, "rgb");
  sensor_msgs::PointCloud2Iterator<float> iter_class(output, "class");

  for (std::size_t i = 0; i < points.size(); ++i,
    ++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_rgb, ++iter_class)
  {
    *iter_x = points[i].x;
    *iter_y = points[i].y;
    *iter_z = points[i].z;
    *iter_intensity = points[i].intensity;

    const auto class_index = static_cast<std::size_t>(std::max(0.0F, classes[i]));
    const auto & color = class_index < kClassColors.size() ?
      kClassColors[class_index] : kClassColors.front();
    *iter_rgb = PackRgb(color);
    *iter_class = classes[i];
  }

  return output;
}

bool ContainsClass(const std::vector<std::int64_t> & class_ids, float point_class)
{
  const auto class_id = static_cast<std::int64_t>(std::llround(point_class));
  return std::find(class_ids.begin(), class_ids.end(), class_id) != class_ids.end();
}

sensor_msgs::msg::LaserScan BuildLaserScan(
  const std::vector<PointXYZI> & points,
  const std::vector<float> & classes,
  const std::vector<std::int64_t> & scan_class_ids,
  const std_msgs::msg::Header & header,
  float angle_min,
  float angle_max,
  float angle_increment,
  float range_min,
  float range_max,
  bool use_inf,
  float inf_epsilon)
{
  sensor_msgs::msg::LaserScan scan;
  scan.header = header;
  scan.angle_min = angle_min;
  scan.angle_max = angle_max;
  scan.angle_increment = angle_increment;
  scan.time_increment = 0.0F;
  scan.scan_time = 0.0F;
  scan.range_min = range_min;
  scan.range_max = range_max;

  const auto ranges_size = static_cast<std::size_t>(
    std::ceil((scan.angle_max - scan.angle_min) / scan.angle_increment));
  scan.ranges.assign(
    ranges_size,
    use_inf ? std::numeric_limits<float>::infinity() : scan.range_max + inf_epsilon);

  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!ContainsClass(scan_class_ids, classes[i])) {
      continue;
    }

    const auto range = static_cast<float>(std::hypot(points[i].x, points[i].y));
    if (range < scan.range_min || range > scan.range_max) {
      continue;
    }

    const auto angle = static_cast<float>(std::atan2(points[i].y, points[i].x));
    if (angle < scan.angle_min || angle > scan.angle_max) {
      continue;
    }

    const auto index = static_cast<std::size_t>((angle - scan.angle_min) / scan.angle_increment);
    if (index >= scan.ranges.size()) {
      continue;
    }

    if (range < scan.ranges[index]) {
      scan.ranges[index] = range;
    }
  }

  return scan;
}

std::string BuildObjectsJson(
  const std::vector<PointXYZI> & points,
  const std::vector<float> & classes,
  const std::vector<std::int64_t> & obstacle_class_ids,
  float cell_size,
  int min_points_per_object)
{
  struct Bucket
  {
    int count {0};
    float sum_x {0.0F};
    float sum_y {0.0F};
  };

  std::unordered_map<std::string, Bucket> buckets;
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!ContainsClass(obstacle_class_ids, classes[i])) {
      continue;
    }

    const auto cell_x = static_cast<int>(std::floor(points[i].x / cell_size));
    const auto cell_y = static_cast<int>(std::floor(points[i].y / cell_size));
    const auto key = std::to_string(cell_x) + ":" + std::to_string(cell_y);
    auto & bucket = buckets[key];
    bucket.count += 1;
    bucket.sum_x += points[i].x;
    bucket.sum_y += points[i].y;
  }

  std::ostringstream stream;
  stream << "[";
  bool first = true;
  for (const auto & [_, bucket] : buckets) {
    if (bucket.count < min_points_per_object) {
      continue;
    }

    if (!first) {
      stream << ",";
    }
    first = false;
    stream << "{\"x_m\":"
           << (bucket.sum_x / static_cast<float>(bucket.count))
           << ",\"y_m\":"
           << (bucket.sum_y / static_cast<float>(bucket.count))
           << ",\"kind\":\"obstacle\"}";
  }
  stream << "]";
  return stream.str();
}

class LivoxLaneDetectionLiveNode : public rclcpp::Node
{
public:
  LivoxLaneDetectionLiveNode()
  : Node("livox_lane_detection_live")
  {
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::vector<std::string>>("lidar_ids", {"6"});
    this->declare_parameter<bool>("publish_colored_cloud", true);
    this->declare_parameter<bool>("publish_lane_scan", true);
    this->declare_parameter<bool>("publish_obstacles", true);
    this->declare_parameter<std::string>("output_frame", "");
    this->declare_parameter<std::vector<std::int64_t>>("scan_class_ids", std::vector<std::int64_t> {});
    this->declare_parameter<std::vector<std::int64_t>>("obstacle_class_ids", std::vector<std::int64_t> {});
    this->declare_parameter<double>("scan_angle_min", -M_PI);
    this->declare_parameter<double>("scan_angle_max", M_PI);
    this->declare_parameter<double>("scan_angle_increment", M_PI / 180.0);
    this->declare_parameter<double>("scan_range_min", 0.0);
    this->declare_parameter<double>("scan_range_max", 120.0);
    this->declare_parameter<bool>("scan_use_inf", true);
    this->declare_parameter<double>("scan_inf_epsilon", 1.0);
    this->declare_parameter<double>("obstacle_cell_size", 0.75);
    this->declare_parameter<int>("obstacle_min_points", 8);
    this->declare_parameter<bool>("publish_timing_log", true);
    this->declare_parameter<double>("timing_log_interval_sec", 2.0);
    this->declare_parameter<double>("max_process_rate_hz", 0.0);
    this->declare_parameter<std::string>("device", "cpu");

    const auto model_path = this->get_parameter("model_path").as_string();
    const auto requested_device = this->get_parameter("device").as_string();
    lidar_ids_ = this->get_parameter("lidar_ids").as_string_array();
    publish_colored_cloud_ = this->get_parameter("publish_colored_cloud").as_bool();
    publish_lane_scan_ = this->get_parameter("publish_lane_scan").as_bool();
    publish_obstacles_ = this->get_parameter("publish_obstacles").as_bool();
    output_frame_ = this->get_parameter("output_frame").as_string();
    scan_class_ids_ = this->get_parameter("scan_class_ids").as_integer_array();
    obstacle_class_ids_ = this->get_parameter("obstacle_class_ids").as_integer_array();
    scan_angle_min_ = static_cast<float>(this->get_parameter("scan_angle_min").as_double());
    scan_angle_max_ = static_cast<float>(this->get_parameter("scan_angle_max").as_double());
    scan_angle_increment_ = static_cast<float>(this->get_parameter("scan_angle_increment").as_double());
    scan_range_min_ = static_cast<float>(this->get_parameter("scan_range_min").as_double());
    scan_range_max_ = static_cast<float>(this->get_parameter("scan_range_max").as_double());
    scan_use_inf_ = this->get_parameter("scan_use_inf").as_bool();
    scan_inf_epsilon_ = static_cast<float>(this->get_parameter("scan_inf_epsilon").as_double());
    obstacle_cell_size_ = static_cast<float>(this->get_parameter("obstacle_cell_size").as_double());
    obstacle_min_points_ = this->get_parameter("obstacle_min_points").as_int();
    publish_timing_log_ = this->get_parameter("publish_timing_log").as_bool();
    timing_log_interval_ms_ = std::max(
      100,
      static_cast<int>(
        std::round(this->get_parameter("timing_log_interval_sec").as_double() * 1000.0)));
    max_process_rate_hz_ = std::max(0.0, this->get_parameter("max_process_rate_hz").as_double());
    inference_device_ = ResolveTorchDevice(requested_device, this->get_logger());

    if (model_path.empty()) {
      throw std::runtime_error("Parameter model_path must be set.");
    }

    range_settings_ = GetBvRangeSettings(lidar_ids_);

    try {
      module_ = torch::jit::load(model_path);
      module_.eval();
      try {
        module_.to(inference_device_);
      } catch (const c10::Error & error) {
        if (inference_device_.type() != torch::kCPU) {
          RCLCPP_WARN(
            this->get_logger(),
            "Failed to move livox_lane_detection model to '%s', falling back to CPU: %s",
            requested_device.c_str(), error.what());
          inference_device_ = torch::Device(torch::kCPU);
          module_.to(inference_device_);
        } else {
          throw;
        }
      }
    } catch (const c10::Error & error) {
      throw std::runtime_error(
              "Failed to load TorchScript model '" + model_path + "': " + error.what());
    }
    RCLCPP_INFO(
      this->get_logger(), "livox_lane_detection loaded model=%s device=%s max_process_rate_hz=%.2f",
      model_path.c_str(), inference_device_.str().c_str(), max_process_rate_hz_);

    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud_in",
      rclcpp::SensorDataQoS(),
      std::bind(&LivoxLaneDetectionLiveNode::CloudCallback, this, std::placeholders::_1));

    if (publish_colored_cloud_) {
      colored_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "points_with_class",
        rclcpp::SensorDataQoS());
    }
    if (publish_lane_scan_) {
      lane_scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("lane_scan", 10);
    }
    if (publish_obstacles_) {
      obstacle_pub_ = this->create_publisher<std_msgs::msg::String>("detected_objects", 10);
    }
  }

private:
  void CloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    const auto total_start = SteadyClock::now();
    if (max_process_rate_hz_ > 0.0 && has_last_process_start_) {
      const auto elapsed_sec =
        std::chrono::duration<double>(total_start - last_process_start_).count();
      if (elapsed_sec < (1.0 / max_process_rate_hz_)) {
        return;
      }
    }
    last_process_start_ = total_start;
    has_last_process_start_ = true;

    auto points = ReadPointsFromCloud(*msg);
    const auto read_done = SteadyClock::now();
    if (points.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Received empty or unsupported point cloud.");
      return;
    }

    const auto bv_data = ProduceBvData(points, common_settings_, range_settings_);
    const auto bv_done = SteadyClock::now();

    if (bv_data.width <= 0 || bv_data.height <= 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "BV tensor size is invalid. Check lidar_ids/range settings.");
      return;
    }

    std::vector<float> point_classes;
    double inference_ms = 0.0;
    double classify_ms = 0.0;
    {
      std::lock_guard<std::mutex> lock(module_mutex_);
      const auto inference_start = SteadyClock::now();
      const auto label_map = RunInference(module_, bv_data, inference_device_);
      const auto inference_done = SteadyClock::now();
      point_classes = GetPointsClassFromBv(
        points, label_map, common_settings_, range_settings_, bv_data.width, bv_data.height);
      const auto classify_done = SteadyClock::now();
      inference_ms = MillisecondsBetween(inference_start, inference_done);
      classify_ms = MillisecondsBetween(inference_done, classify_done);
    }

    const auto colored_start = SteadyClock::now();
    if (publish_colored_cloud_) {
      auto header = msg->header;
      if (!output_frame_.empty()) {
        header.frame_id = output_frame_;
      }
      colored_cloud_pub_->publish(BuildColoredCloud(points, point_classes, header));
    }
    const auto colored_done = SteadyClock::now();

    const auto scan_start = SteadyClock::now();
    if (publish_lane_scan_ && lane_scan_pub_ != nullptr && !scan_class_ids_.empty()) {
      auto header = msg->header;
      if (!output_frame_.empty()) {
        header.frame_id = output_frame_;
      }
      lane_scan_pub_->publish(BuildLaserScan(
          points,
          point_classes,
          scan_class_ids_,
          header,
          scan_angle_min_,
          scan_angle_max_,
          scan_angle_increment_,
          scan_range_min_,
          scan_range_max_,
          scan_use_inf_,
          scan_inf_epsilon_));
    }
    const auto scan_done = SteadyClock::now();

    const auto objects_start = SteadyClock::now();
    if (publish_obstacles_ && obstacle_pub_ != nullptr && !obstacle_class_ids_.empty()) {
      std_msgs::msg::String objects_msg;
      objects_msg.data = BuildObjectsJson(
        points, point_classes, obstacle_class_ids_, obstacle_cell_size_, obstacle_min_points_);
      obstacle_pub_->publish(objects_msg);
    }
    const auto objects_done = SteadyClock::now();

    if (publish_timing_log_) {
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), timing_log_interval_ms_,
        "livox_lane_detection timing total_ms=%.2f points=%zu bv=%dx%d read_ms=%.2f "
        "bv_ms=%.2f inference_ms=%.2f classify_ms=%.2f colored_ms=%.2f scan_ms=%.2f "
        "objects_ms=%.2f device=%s frame=%s",
        MillisecondsBetween(total_start, objects_done),
        points.size(),
        bv_data.width,
        bv_data.height,
        MillisecondsBetween(total_start, read_done),
        MillisecondsBetween(read_done, bv_done),
        inference_ms,
        classify_ms,
        MillisecondsBetween(colored_start, colored_done),
        MillisecondsBetween(scan_start, scan_done),
        MillisecondsBetween(objects_start, objects_done),
        inference_device_.str().c_str(),
        msg->header.frame_id.c_str());
    }
  }

  BvCommonSettings common_settings_ {};
  BvRangeSettings range_settings_ {};
  std::vector<std::string> lidar_ids_;
  bool publish_colored_cloud_ {true};
  bool publish_lane_scan_ {true};
  bool publish_obstacles_ {true};
  std::string output_frame_;
  std::vector<std::int64_t> scan_class_ids_;
  std::vector<std::int64_t> obstacle_class_ids_;
  float scan_angle_min_ {-static_cast<float>(M_PI)};
  float scan_angle_max_ {static_cast<float>(M_PI)};
  float scan_angle_increment_ {static_cast<float>(M_PI / 180.0)};
  float scan_range_min_ {0.0F};
  float scan_range_max_ {120.0F};
  bool scan_use_inf_ {true};
  float scan_inf_epsilon_ {1.0F};
  float obstacle_cell_size_ {0.75F};
  int obstacle_min_points_ {8};
  bool publish_timing_log_ {true};
  int timing_log_interval_ms_ {2000};
  double max_process_rate_hz_ {0.0};
  bool has_last_process_start_ {false};
  SteadyClock::time_point last_process_start_ {};
  torch::Device inference_device_ {torch::kCPU};
  torch::jit::script::Module module_;
  std::mutex module_mutex_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr lane_scan_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr obstacle_pub_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<LivoxLaneDetectionLiveNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("livox_lane_detection_live"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
