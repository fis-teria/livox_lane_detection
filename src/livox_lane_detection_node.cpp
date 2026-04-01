#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace fs = std::filesystem;

namespace
{

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

struct Parameters
{
  std::array<float, 16> p_world {};
};

struct BvTensor
{
  int height {0};
  int width {0};
  std::vector<float> data {};
};

using PointMap = std::unordered_map<std::string, std::vector<PointXYZI>>;
using ClassMap = std::unordered_map<std::string, std::vector<float>>;
using ParameterMap = std::unordered_map<std::string, Parameters>;

constexpr int kPointCloudHeaderLines = 11;

std::string Trim(const std::string & input)
{
  const auto begin = std::find_if_not(input.begin(), input.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
  const auto end = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::vector<float> ParseFloatLine(const std::string & line)
{
  std::istringstream stream(line);
  std::vector<float> values;
  float value = 0.0F;
  while (stream >> value) {
    values.push_back(value);
  }
  return values;
}

std::array<float, 16> LoadMatrix4(const std::array<std::string, 4> & rows)
{
  std::array<float, 16> matrix {};
  for (std::size_t row = 0; row < rows.size(); ++row) {
    const auto values = ParseFloatLine(rows[row]);
    if (values.size() != 4U) {
      throw std::runtime_error("Invalid 4x4 matrix row in parameter file.");
    }
    for (std::size_t col = 0; col < 4U; ++col) {
      matrix[row * 4U + col] = values[col];
    }
  }
  return matrix;
}

PointXYZI TransformPoint(const PointXYZI & point, const std::array<float, 16> & matrix)
{
  PointXYZI output = point;
  output.x = matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3];
  output.y = matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7];
  output.z = matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11];
  return output;
}

std::vector<PointXYZI> ReadPcdFile(const fs::path & file_name)
{
  std::ifstream input(file_name);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open PCD file: " + file_name.string());
  }

  std::string line;
  for (int i = 0; i < kPointCloudHeaderLines; ++i) {
    if (!std::getline(input, line)) {
      throw std::runtime_error("Unexpected end of PCD header: " + file_name.string());
    }
  }

  std::vector<PointXYZI> points;
  points.reserve(65536);
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream stream(line);
    PointXYZI point;
    stream >> point.x >> point.y >> point.z >> point.intensity;
    if (!stream.fail()) {
      points.push_back(point);
    }
  }
  return points;
}

void WritePointsToFile(
  const std::vector<PointXYZI> & points, const std::vector<float> & classes, const fs::path & filename)
{
  std::ofstream output(filename);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to write PCD file: " + filename.string());
  }

  output << "# .PCD v.7 - Point Cloud Data file format\n";
  output << "VERSION .7\n";
  output << "FIELDS x y z intensity class\n";
  output << "SIZE 4 4 4 4 4\n";
  output << "TYPE F F F F F\n";
  output << "COUNT 1 1 1 1 1\n";
  output << "WIDTH " << points.size() << "\n";
  output << "HEIGHT 1\n";
  output << "VIEWPOINT 0 0 0 1 0 0 0\n";
  output << "POINTS " << points.size() << "\n";
  output << "DATA ascii\n";
  output << std::fixed << std::setprecision(3);

  for (std::size_t i = 0; i < points.size(); ++i) {
    output << points[i].x << ' ' << points[i].y << ' ' << points[i].z << ' '
           << points[i].intensity << ' ' << classes[i] << '\n';
  }
}

std::vector<fs::path> GlobSorted(const fs::path & directory, const std::string & extension)
{
  std::vector<fs::path> files;
  if (!fs::exists(directory)) {
    return files;
  }
  for (const auto & entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == extension) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

struct DatasetFrame
{
  std::unordered_map<std::string, fs::path> pointcloud_names;
};

std::vector<DatasetFrame> GetTestDataList(
  const fs::path & folder, const std::vector<std::string> & lidar_ids)
{
  std::vector<DatasetFrame> frames;
  if (lidar_ids.empty()) {
    return frames;
  }

  const auto first_lidar_files = GlobSorted(folder / "lidar" / lidar_ids.front(), ".pcd");
  frames.resize(first_lidar_files.size());

  for (const auto & lidar_id : lidar_ids) {
    const auto files = GlobSorted(folder / "lidar" / lidar_id, ".pcd");
    for (std::size_t i = 0; i < std::min(files.size(), frames.size()); ++i) {
      frames[i].pointcloud_names[lidar_id] = files[i];
    }
  }

  return frames;
}

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

Parameters DefaultParametersForLidar(const std::string & lidar_id)
{
  static const std::unordered_map<std::string, std::array<float, 16>> kDefaultPWorld = {
    {"1", {0.999972F, 0.0F, 0.00750485F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
           -0.00750485F, 0.0F, 0.999972F, 1.909F, 0.0F, 0.0F, 0.0F, 1.0F}},
    {"2", {0.338535F, -0.939803F, 0.0465286F, -0.001F, 0.929107F, 0.341685F, 0.141463F, 0.338F,
           -0.148845F, -0.00465988F, 0.98885F, 1.909F, 0.0F, 0.0F, 0.0F, 1.0F}},
    {"3", {-0.742701F, -0.656677F, -0.131041F, -1.109F, 0.64927F, -0.754084F, 0.0990242F, 0.15F,
           -0.163842F, -0.0115354F, 0.986419F, 1.909F, 0.0F, 0.0F, 0.0F, 1.0F}},
    {"4", {-0.770801F, 0.62401F, -0.128361F, -1.2F, -0.617308F, -0.781369F, -0.0916196F, -0.149F,
           -0.157469F, 0.00861766F, 0.987486F, 1.909F, 0.0F, 0.0F, 0.0F, 1.0F}},
    {"5", {0.33825F, 0.939574F, 0.0528062F, -0.009F, -0.929838F, 0.342329F, -0.134952F, -0.301F,
           -0.144874F, -0.00345383F, 0.989444F, 1.909F, 0.0F, 0.0F, 0.0F, 1.0F}},
    {"6", {0.999931F, -0.000861352F, 0.0117086F, -0.037F, 0.00104713F, 0.999874F, -0.0158696F, 0.019F,
           -0.0116934F, 0.0158807F, 0.999806F, 1.909F, 0.0F, 0.0F, 0.0F, 1.0F}},
  };

  auto iter = kDefaultPWorld.find(lidar_id);
  if (iter == kDefaultPWorld.end()) {
    throw std::runtime_error("Unsupported lidar id: " + lidar_id);
  }
  return Parameters {iter->second};
}

ParameterMap ReadSelectedParameters(
  const fs::path & folder, const std::vector<std::string> & lidar_ids)
{
  ParameterMap result;
  for (const auto & lidar_id : lidar_ids) {
    const auto parameter_file = folder / "parameter" / lidar_id / "config.txt";
    if (!fs::exists(parameter_file)) {
      result[lidar_id] = DefaultParametersForLidar(lidar_id);
      continue;
    }

    std::ifstream input(parameter_file);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open parameter file: " + parameter_file.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
      lines.push_back(Trim(line));
    }
    if (lines.size() < 19U) {
      throw std::runtime_error("Parameter file is too short: " + parameter_file.string());
    }

    result[lidar_id] = Parameters {
      LoadMatrix4({lines[15], lines[16], lines[17], lines[18]})
    };
  }

  return result;
}

PointMap ReadSelectedPoints(const DatasetFrame & frame)
{
  PointMap result;
  for (const auto & [lidar_id, path] : frame.pointcloud_names) {
    result.emplace(lidar_id, ReadPcdFile(path));
  }
  return result;
}

PointMap ProjectPointsToWorld(const PointMap & points_input_set, const ParameterMap & parameters)
{
  PointMap points_output_set;
  for (const auto & [lidar_id, input_points] : points_input_set) {
    const auto param_iter = parameters.find(lidar_id);
    if (param_iter == parameters.end()) {
      throw std::runtime_error("Missing world transform for lidar " + lidar_id);
    }

    auto transformed = input_points;
    for (auto & point : transformed) {
      point = TransformPoint(point, param_iter->second.p_world);
    }
    points_output_set.emplace(lidar_id, std::move(transformed));
  }
  return points_output_set;
}

std::vector<PointXYZI> MergePoints(const PointMap & points_input_set)
{
  std::size_t total_points = 0U;
  for (const auto & [_, points] : points_input_set) {
    total_points += points.size();
  }

  std::vector<PointXYZI> merged;
  merged.reserve(total_points);
  for (const auto & [_, points] : points_input_set) {
    merged.insert(merged.end(), points.begin(), points.end());
  }
  return merged;
}

BvTensor ProduceBvData(
  const std::vector<PointXYZI> & points, const BvCommonSettings & common,
  const BvRangeSettings & range)
{
  const int bv_width = static_cast<int>((range.right_distance + range.left_distance) *
    common.width_resolution_train);
  const int bv_height = static_cast<int>((range.max_distance - range.min_distance) *
    common.distance_resolution_train);

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

    const int image_x = static_cast<int>(-point.y * common.width_resolution_train + left_distance_px);
    const int image_y = static_cast<int>(max_distance_px - point.x * common.distance_resolution_train);

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
          std::max(tensor.data[static_cast<std::size_t>(bv_width * bv_height) + index], shifted_z);
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

std::vector<std::int64_t> RunInference(torch::jit::script::Module & module, const BvTensor & bv_data)
{
  torch::NoGradGuard no_grad;

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto input = torch::from_blob(
    const_cast<float *>(bv_data.data.data()),
    {1, 2, bv_data.height, bv_data.width},
    options).clone();

  auto output = ForwardModule(module, input);
  auto label_map = output.argmax(1).squeeze().to(torch::kCPU).contiguous();

  const auto numel = label_map.numel();
  std::vector<std::int64_t> result(static_cast<std::size_t>(numel));
  std::memcpy(result.data(), label_map.data_ptr<std::int64_t>(), sizeof(std::int64_t) * static_cast<std::size_t>(numel));
  return result;
}

ClassMap GetPointsClassFromBv(
  const PointMap & points_input_set, const std::vector<std::int64_t> & bv_label_map,
  const BvCommonSettings & common, const BvRangeSettings & range, int bv_width, int bv_height)
{
  ClassMap points_class_set;

  const float max_distance_px = range.max_distance * common.distance_resolution_train;
  const float left_distance_px = range.left_distance * common.width_resolution_train;

  for (const auto & [lidar_id, points] : points_input_set) {
    std::vector<float> point_classes(points.size(), 0.0F);
    for (std::size_t i = 0; i < points.size(); ++i) {
      const float shifted_z = points[i].z + common.train_height_shift;
      if (shifted_z < common.shifted_min_height || shifted_z > common.shifted_max_height) {
        continue;
      }

      const int image_x = static_cast<int>(-points[i].y * common.width_resolution_train + left_distance_px);
      const int image_y = static_cast<int>(max_distance_px - points[i].x * common.distance_resolution_train);

      if (image_x >= 0 && image_x < bv_width && image_y >= 0 && image_y < bv_height) {
        point_classes[i] = static_cast<float>(bv_label_map[static_cast<std::size_t>(image_y * bv_width + image_x)]);
      }
    }
    points_class_set.emplace(lidar_id, std::move(point_classes));
  }

  return points_class_set;
}

void CopyParameterFileIfExists(const fs::path & input_path, const fs::path & output_path)
{
  if (!fs::exists(input_path)) {
    return;
  }
  fs::create_directories(output_path.parent_path());
  if (!fs::exists(output_path)) {
    fs::copy_file(input_path, output_path);
  }
}

void OutputPointsWithClass(
  const PointMap & points_set, const ClassMap & points_class_set, const fs::path & subfolder_output_path,
  const DatasetFrame & frame, const fs::path & source_subfolder)
{
  const auto output_path_lidar = subfolder_output_path / "lidar";
  const auto output_path_parameter = subfolder_output_path / "parameter";
  fs::create_directories(output_path_lidar);
  fs::create_directories(output_path_parameter);

  for (const auto & [lidar_id, input_path] : frame.pointcloud_names) {
    const auto output_folder = output_path_lidar / lidar_id;
    fs::create_directories(output_folder);
    WritePointsToFile(
      points_set.at(lidar_id), points_class_set.at(lidar_id), output_folder / input_path.filename());

    const auto parameter_input = source_subfolder / "parameter" / lidar_id / "config.txt";
    const auto parameter_output = output_path_parameter / lidar_id / "config.txt";
    CopyParameterFileIfExists(parameter_input, parameter_output);
  }
}

class LivoxLaneDetectionNode : public rclcpp::Node
{
public:
  LivoxLaneDetectionNode()
  : Node("livox_lane_detection")
  {
    this->declare_parameter<std::vector<std::string>>("lidar_ids", {"1", "2", "3", "4", "5", "6"});
    this->declare_parameter<std::string>("test_data_folder", "");
    this->declare_parameter<std::string>("output_folder", "");
    this->declare_parameter<std::string>("model_path", "");
  }

  int Run()
  {
    const auto lidar_ids = this->get_parameter("lidar_ids").as_string_array();
    const auto test_data_folder = this->get_parameter("test_data_folder").as_string();
    const auto output_folder = this->get_parameter("output_folder").as_string();
    const auto model_path = this->get_parameter("model_path").as_string();

    if (test_data_folder.empty() || output_folder.empty() || model_path.empty()) {
      RCLCPP_ERROR(
        get_logger(),
        "Parameters test_data_folder, output_folder, and model_path must all be set.");
      return 1;
    }

    BvCommonSettings common_settings;
    const auto range_settings = GetBvRangeSettings(lidar_ids);

    torch::jit::script::Module module;
    try {
      module = torch::jit::load(model_path);
      module.eval();
    } catch (const c10::Error & error) {
      RCLCPP_ERROR(get_logger(), "Failed to load TorchScript model '%s': %s", model_path.c_str(), error.what());
      return 1;
    }

    std::vector<fs::path> subfolders;
    for (const auto & entry : fs::directory_iterator(test_data_folder)) {
      if (entry.is_directory()) {
        subfolders.push_back(entry.path());
      }
    }
    std::sort(subfolders.begin(), subfolders.end());

    for (const auto & subfolder : subfolders) {
      RCLCPP_INFO(get_logger(), "Processing %s", subfolder.c_str());
      const auto frames = GetTestDataList(subfolder, lidar_ids);
      const auto parameters = ReadSelectedParameters(subfolder, lidar_ids);
      const auto subfolder_output = fs::path(output_folder) / subfolder.filename();
      fs::create_directories(subfolder_output);

      for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        RCLCPP_INFO(
          get_logger(), "Frame %zu / %zu", frame_index + 1U, frames.size());

        const auto points_input = ReadSelectedPoints(frames[frame_index]);
        const auto points_transformed = ProjectPointsToWorld(points_input, parameters);
        const auto points_merged = MergePoints(points_transformed);
        const auto bv_data = ProduceBvData(points_merged, common_settings, range_settings);
        const auto label_map = RunInference(module, bv_data);
        const auto point_classes = GetPointsClassFromBv(
          points_transformed, label_map, common_settings, range_settings, bv_data.width, bv_data.height);
        OutputPointsWithClass(points_input, point_classes, subfolder_output, frames[frame_index], subfolder);
      }
    }

    return 0;
  }
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LivoxLaneDetectionNode>();
  const int result = node->Run();
  rclcpp::shutdown();
  return result;
}
