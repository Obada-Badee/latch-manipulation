# Importable Python module for the yolov8_obb package.
#
# The executable nodes live in scripts/ and are installed to lib/yolov8_obb by
# CMakeLists.txt. This directory is different: it is installed as a real Python
# package via ament_python_install_package(), so anything here can be imported
# as `from yolov8_obb.projection import ...` by any node - and by UI/bolt_selector.py,
# which runs outside the install tree.
