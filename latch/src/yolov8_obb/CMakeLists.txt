cmake_minimum_required(VERSION 3.8)
project(yolov8_obb)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install yolov8_obb/ as a real importable Python package (-> lib/python3.*/
# site-packages/yolov8_obb). The nodes in scripts/ are installed as executables
# into lib/${PROJECT_NAME}, which is NOT on PYTHONPATH - so shared code such as
# projection.py has to live here instead to be importable by the nodes, by
# tools/synth_dataset, and by UI/bolt_selector.py, which runs outside the
# install tree entirely.
ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    scripts
    launch
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
  scripts/yolov8_obb_publisher.py
  scripts/world_model.py
  scripts/debug_viewer.py
  launch/yolov8_obb.launch.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
