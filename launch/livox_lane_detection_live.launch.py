from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    livox_share = FindPackageShare("livox_ros_driver2")
    lane_share = FindPackageShare("livox_lane_detection")

    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    model_path = LaunchConfiguration("model_path")
    colored_cloud_topic = LaunchConfiguration("colored_cloud_topic")
    lane_scan_topic = LaunchConfiguration("lane_scan_topic")
    objects_topic = LaunchConfiguration("objects_topic")
    output_frame = LaunchConfiguration("output_frame")
    launch_livox_driver = LaunchConfiguration("launch_livox_driver")
    torch_lib_path = LaunchConfiguration("torch_lib_path")
    publish_colored_cloud = LaunchConfiguration("publish_colored_cloud")
    publish_lane_scan = LaunchConfiguration("publish_lane_scan")
    publish_obstacles = LaunchConfiguration("publish_obstacles")
    publish_timing_log = LaunchConfiguration("publish_timing_log")
    timing_log_interval_sec = LaunchConfiguration("timing_log_interval_sec")
    max_process_rate_hz = LaunchConfiguration("max_process_rate_hz")
    device = LaunchConfiguration("device")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pointcloud_topic",
                default_value="/livox/lidar",
                description="Livox HAP PointCloud2 topic",
            ),
            DeclareLaunchArgument(
                "colored_cloud_topic",
                default_value="/livox/lane_detection/points_with_class",
                description="Colored classification point cloud topic",
            ),
            DeclareLaunchArgument(
                "lane_scan_topic",
                default_value="/livox/lane_detection/scan",
                description="Lane and road-edge LaserScan topic for GUI",
            ),
            DeclareLaunchArgument(
                "objects_topic",
                default_value="/livox/lane_detection/objects",
                description="Obstacle JSON topic for GUI",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [lane_share, "model", "livox_lane_det.ts"]
                ),
                description="TorchScript model path",
            ),
            DeclareLaunchArgument(
                "output_frame",
                default_value="",
                description="Optional output frame override for visualization",
            ),
            DeclareLaunchArgument(
                "launch_livox_driver",
                default_value="false",
                description="Whether to also launch livox_ros_driver2 HAP driver",
            ),
            DeclareLaunchArgument(
                "torch_lib_path",
                default_value="/opt/libtorch/lib",
                description=(
                    "libtorch runtime library directory to prepend for the C++ node. "
                    "CUDA builds can pass a colon-separated torch/nvjitlink library path."
                ),
            ),
            DeclareLaunchArgument(
                "publish_colored_cloud",
                default_value="false",
                description="Publish classified PointCloud2 output.",
            ),
            DeclareLaunchArgument(
                "publish_lane_scan",
                default_value="true",
                description="Publish lane/road-edge LaserScan output.",
            ),
            DeclareLaunchArgument(
                "publish_obstacles",
                default_value="false",
                description="Publish obstacle JSON output.",
            ),
            DeclareLaunchArgument(
                "publish_timing_log",
                default_value="true",
                description="Log per-stage processing timings.",
            ),
            DeclareLaunchArgument(
                "timing_log_interval_sec",
                default_value="2.0",
                description="Minimum interval between timing log lines.",
            ),
            DeclareLaunchArgument(
                "max_process_rate_hz",
                default_value="5.0",
                description="Maximum lane-detection processing rate. 0 disables throttling.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cuda:0",
                description="Torch inference device. Falls back to CPU when CUDA is unavailable.",
            ),
            Node(
                package="livox_lane_detection",
                executable="livox_lane_detection_live_node",
                name="livox_lane_detection_live",
                output="screen",
                additional_env={
                    "LD_LIBRARY_PATH": [
                        torch_lib_path,
                        ":",
                        EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
                    ],
                },
                parameters=[
                    {
                        "model_path": model_path,
                        "lidar_ids": ["6"],
                        "publish_colored_cloud": ParameterValue(
                            publish_colored_cloud, value_type=bool
                        ),
                        "publish_lane_scan": ParameterValue(
                            publish_lane_scan, value_type=bool
                        ),
                        "publish_obstacles": ParameterValue(
                            publish_obstacles, value_type=bool
                        ),
                        "publish_timing_log": ParameterValue(
                            publish_timing_log, value_type=bool
                        ),
                        "timing_log_interval_sec": ParameterValue(
                            timing_log_interval_sec, value_type=float
                        ),
                        "max_process_rate_hz": ParameterValue(
                            max_process_rate_hz, value_type=float
                        ),
                        "device": device,
                        "output_frame": output_frame,
                        # These IDs should be tuned to your trained model's label map.
                        "scan_class_ids": [3, 29],
                        "obstacle_class_ids": [15, 23, 25, 27],
                    }
                ],
                remappings=[
                    ("cloud_in", pointcloud_topic),
                    ("points_with_class", colored_cloud_topic),
                    ("lane_scan", lane_scan_topic),
                    ("detected_objects", objects_topic),
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([livox_share, "launch_ROS2", "msg_HAP_launch.py"])
                ),
                condition=IfCondition(launch_livox_driver),
            ),
        ]
    )
