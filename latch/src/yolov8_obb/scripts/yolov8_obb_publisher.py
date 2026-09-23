#!/usr/bin/env python3
"""Run the YOLOv8-OBB detector on /image_raw and publish oriented boxes.

Publishes:
    /Yolov8_Inference   yolov8_msgs/Yolov8Inference   class + 4 corner points per detection
    /inference_result   sensor_msgs/Image             annotated frame, for debugging

The detector reports *type* only (cube / bolt / cylinder). Colour is deliberately
not a class - it is derived downstream in world_model from pixels, which keeps the
class count (and therefore the training set) small. See tools/synth_dataset/.
"""

import copy
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from yolov8_msgs.msg import InferenceResult, Yolov8Inference

bridge = CvBridge()


class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        # Was hardcoded to $HOME/moveit2_obb/src/... which forced the workspace to
        # live at one exact path. Resolve from the installed share dir instead, with
        # a parameter override for pointing at a freshly trained checkpoint.
        default_weights = os.path.join(
            get_package_share_directory('yolov8_obb'), 'scripts', 'best.pt')
        self.declare_parameter('weights', default_weights)
        self.declare_parameter('conf', 0.50)
        self.declare_parameter('image_topic', '/image_raw')
        # Class-agnostic NMS, on by default. YOLO's default NMS is per-class, so
        # two boxes on the *same* object never suppress each other if the model
        # labelled them differently. Observed live: one green cube reported both
        # as a `cube` and as a `cylinder`, which world_model then tracked as two
        # separate objects with two ids - so the planner would see a phantom
        # neighbour and could try to pick something that is not there.
        #
        # Here a physical object has exactly one class, so suppressing across
        # classes is not a heuristic, it is the truth about the scene.
        self.declare_parameter('agnostic_nms', True)
        # Inference resolution. 0 means "whatever this checkpoint was trained
        # at", read from the checkpoint itself.
        #
        # This must match training or the model sees objects at the wrong scale
        # and quietly gets worse - no error, just degraded detections. Making it
        # a hand-set constant would mean every weight swap is a chance to forget,
        # and the symptom would look like a bad model rather than a bad flag.
        self.declare_parameter('imgsz', 0)

        # An empty string means "use the installed default". The launch file
        # passes '' when the operator has not overridden the checkpoint, and a
        # LaunchConfiguration cannot be conditionally omitted from a parameter
        # dict - so the sentinel is resolved here instead.
        weights = self.get_parameter('weights').value or default_weights
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f'YOLO weights not found: {weights}. Train a checkpoint with '
                f'tools/synth_dataset/train.py --install, then rebuild '
                f'yolov8_obb so the install space picks it up.')
        self.conf = float(self.get_parameter('conf').value)
        self.agnostic_nms = bool(self.get_parameter('agnostic_nms').value)
        self.model = YOLO(weights)
        self.imgsz = int(self.get_parameter('imgsz').value) or self._trained_imgsz()
        self.get_logger().info(
            f'loaded {weights} - classes: {self.model.names}, conf={self.conf}, '
            f'agnostic_nms={self.agnostic_nms}, imgsz={self.imgsz}')

        self.yolov8_inference = Yolov8Inference()

        self.subscription = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.camera_callback,
            10)

        self.yolov8_pub = self.create_publisher(Yolov8Inference, '/Yolov8_Inference', 1)
        self.img_pub = self.create_publisher(Image, '/inference_result', 1)

    def _trained_imgsz(self, fallback: int = 640) -> int:
        """The imgsz this checkpoint was trained at, from the checkpoint.

        Ultralytics stores the training arguments in the weights file, so the
        right value travels with the model instead of having to be remembered
        alongside it. Several attribute paths are tried because the location
        has moved between versions, and a wrong guess here is silent.
        """
        for get in (lambda: self.model.overrides.get('imgsz'),
                    lambda: self.model.ckpt['train_args']['imgsz'],
                    lambda: self.model.model.args['imgsz']):
            try:
                value = get()
            except Exception:                                    # noqa: BLE001
                continue
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
            if isinstance(value, (list, tuple)) and value:
                return int(value[0])
        self.get_logger().warning(
            f'could not read the training imgsz from the checkpoint; using '
            f'{fallback}. If this model was trained at another size its '
            f'accuracy will be silently worse - set the imgsz parameter.')
        return fallback

    def camera_callback(self, data):
        img = bridge.imgmsg_to_cv2(data, 'bgr8')
        results = self.model(img, conf=self.conf, imgsz=self.imgsz,
                             agnostic_nms=self.agnostic_nms, verbose=False)

        self.yolov8_inference.header.frame_id = 'inference'
        # Propagate the source frame's stamp rather than stamping "now", so
        # world_model can tell how stale a detection is.
        self.yolov8_inference.header.stamp = data.header.stamp

        for r in results:
            if r.obb is None:
                self.get_logger().debug('no detections in frame')
                continue
            for box in r.obb:
                result = InferenceResult()
                corners = box.xyxyxyxy[0].to('cpu').detach().numpy().copy()
                result.class_name = self.model.names[int(box.cls)]
                result.coordinates = copy.copy(corners.reshape(1, 8)[0].tolist())
                self.yolov8_inference.yolov8_inference.append(result)

        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

        annotated = results[0].plot()
        self.img_pub.publish(bridge.cv2_to_imgmsg(annotated, 'bgr8'))


def main():
    rclpy.init(args=None)
    node = Camera_subscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
