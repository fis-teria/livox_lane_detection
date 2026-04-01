from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    share_dir = FindPackageShare("livox_lane_detection")
    return LaunchDescription(
        [
            Node(
                package="livox_lane_detection",
                executable="livox_lane_detection_node",
                name="livox_lane_detection",
                output="screen",
                parameters=[
                    {
                        "test_data_folder": PathJoinSubstitution([share_dir, "test_data"]),
                        "output_folder": PathJoinSubstitution(
                            [share_dir, "result", "points_with_class_cpp"]
                        ),
                        "model_path": PathJoinSubstitution(
                            [share_dir, "model", "livox_lane_det.ts"]
                        ),
                        "lidar_ids": ["1", "2", "3", "4", "5", "6"],
                    }
                ],
            )
        ]
    )
