cmake_minimum_required(VERSION 3.8)
project(livox_lane_detection)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

option(LIVOX_LANE_DETECTION_WITH_TORCH "Build the TorchScript C++ inference executable" ON)
option(
  LIVOX_LANE_DETECTION_ALLOW_PYTHON_TORCH
  "Allow building against a Python wheel's torch CMake package. This is usually unsafe for the C++ node."
  OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

if(LIVOX_LANE_DETECTION_WITH_TORCH)
  set(_livox_lane_detection_torch_search_paths "")

  if(DEFINED Torch_DIR AND NOT "${Torch_DIR}" STREQUAL "")
    list(APPEND _livox_lane_detection_torch_search_paths "${Torch_DIR}")
  endif()

  if(DEFINED ENV{Torch_DIR} AND NOT "$ENV{Torch_DIR}" STREQUAL "")
    list(APPEND _livox_lane_detection_torch_search_paths "$ENV{Torch_DIR}")
  endif()

  list(REMOVE_DUPLICATES _livox_lane_detection_torch_search_paths)

  if(_livox_lane_detection_torch_search_paths AND NOT LIVOX_LANE_DETECTION_ALLOW_PYTHON_TORCH)
    foreach(_torch_search_path IN LISTS _livox_lane_detection_torch_search_paths)
      if(_torch_search_path MATCHES "site-packages.*/torch/share/cmake/Torch")
        message(FATAL_ERROR
          "livox_lane_detection was asked to build against Python wheel Torch at "
          "${_torch_search_path}. This often produces runtime symbol lookup errors. "
          "Use /opt/libtorch/share/cmake/Torch or set "
          "LIVOX_LANE_DETECTION_ALLOW_PYTHON_TORCH=ON if you really want this.")
      endif()
    endforeach()
  endif()

  if(_livox_lane_detection_torch_search_paths)
    find_package(Torch QUIET CONFIG PATHS ${_livox_lane_detection_torch_search_paths} NO_DEFAULT_PATH)
  else()
    message(WARNING
      "Torch auto-discovery is disabled for livox_lane_detection to avoid broken system "
      "packages aborting the workspace build. Set Torch_DIR to a valid libtorch/Torch "
      "CMake package path to build livox_lane_detection_node.")
  endif()

  if(Torch_FOUND)
    add_executable(livox_lane_detection_node
      src/livox_lane_detection_node.cpp
    )
    add_executable(livox_lane_detection_live_node
      src/livox_lane_detection_live_node.cpp
    )
    target_compile_features(livox_lane_detection_node PUBLIC cxx_std_17)
    target_compile_features(livox_lane_detection_live_node PUBLIC cxx_std_17)
    target_include_directories(livox_lane_detection_node PRIVATE include)
    target_include_directories(livox_lane_detection_live_node PRIVATE include)
    target_compile_options(livox_lane_detection_node PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_options(livox_lane_detection_live_node PRIVATE ${TORCH_CXX_FLAGS})
    ament_target_dependencies(livox_lane_detection_node rclcpp sensor_msgs std_msgs)
    ament_target_dependencies(livox_lane_detection_live_node rclcpp sensor_msgs std_msgs)
    target_link_libraries(livox_lane_detection_node "${TORCH_LIBRARIES}")
    target_link_libraries(livox_lane_detection_live_node "${TORCH_LIBRARIES}")
    target_compile_definitions(livox_lane_detection_node PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
    target_compile_definitions(livox_lane_detection_live_node PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
    set_property(TARGET livox_lane_detection_node PROPERTY CXX_STANDARD 17)
    set_property(TARGET livox_lane_detection_live_node PROPERTY CXX_STANDARD 17)
    install(TARGETS livox_lane_detection_node livox_lane_detection_live_node
      DESTINATION lib/${PROJECT_NAME}
    )
  else()
    message(WARNING
      "Torch not found. livox_lane_detection_node will not be built. "
      "Set Torch_DIR to a valid libtorch/Torch CMake package path.")
  endif()
endif()

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

install(DIRECTORY model test_data
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

install(FILES
  README.md
  LICENSE
  config.py
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/export_torchscript.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
