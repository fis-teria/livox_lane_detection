from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
                    "Keep this ahead of Python wheel torch libs."
                ),
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
                        "publish_colored_cloud": True,
                        "publish_lane_scan": True,
                        "publish_obstacles": True,
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
