#!/usr/bin/env python3
"""Turn raw OBB detections into a grounded, stable world state.

/Yolov8_Inference gives class + four pixel corners, per frame, with no identity
and no colour. The planner needs something it can refer to in language:
persistent ids, robot-frame coordinates, colour, and grasp affordances.

Three jobs:

  1. Back-project pixels to panda_link0 using yolov8_obb.projection - the same
     module the synthetic dataset generator inverts, so the two cannot drift.

  2. Classify colour from the image, not from the class. Colour is deliberately
     NOT a YOLO class: making it one multiplies classes by colours and forces a
     retrain for every new shade. A hue histogram inside the detected box costs
     nothing and generalises to colours never trained on.

  3. Assign persistent ids by nearest-neighbour matching against the previous
     frame. Without this, "pick up object 3" means a different object each
     frame and the planner cannot act on anything it saw a moment ago.

Publishes /world_state at a steady rate rather than per detection, so the
planner reads a coherent snapshot instead of racing the detector.

Receptacles (the bins) are NOT published here. They are static, they are not
detected, and having perception assert them would invert the dependency - the
planner merges them in from llm_task_planner.scene, which is also what the
skill server uses, so the two cannot disagree about where a bin is.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from yolov8_msgs.msg import DetectedObject, WorldState, Yolov8Inference

from yolov8_obb.projection import (
    CAMERA, DEFAULT_CENTROID_Z, OBJECT_CENTROID_Z, bolt_grasp_pixel,
    grasp_yaw, obb_axes_px, obb_center, obb_long_axis_yaw,
)

bridge = CvBridge()

#: Hue bands in OpenCV's 0..179 scale. Red straddles the wrap point, so it gets
#: two bands. Taken from the ranges used in the Webots project's camera.py, but
#: applied to a masked region rather than to contours found by an
#: approxPolyDP call that was being passed an arc length as its epsilon.
HUE_BANDS: List[Tuple[str, int, int]] = [
    ('red', 0, 10),
    ('yellow', 20, 35),
    ('green', 40, 85),
    ('blue', 90, 130),
    ('red', 170, 179),
]

#: Below these a pixel is grey/black/white and carries no usable hue. The table
#: is a desaturated wood colour, so this also rejects most background bleed.
MIN_SATURATION = 70
MIN_VALUE = 50

#: Largest jump, in metres, that still counts as the same object between frames.
#: Objects on a table are static except while being carried; this only has to
#: beat the detector's jitter.
ID_MATCH_RADIUS = 0.06

#: Frames an object may go unseen before its id is retired. A grasped object is
#: occluded by the gripper for several frames and should keep its identity.
ID_MAX_MISSES = 15

#: Where a table object can possibly be, in link0. A detection that
#: back-projects outside this is discarded.
#:
#: This is not a tuning knob, it is a consistency check. Back-projection assumes
#: the detection lies on the object's centroid plane just above the table. When
#: the detector fires on something that is *not* on the table - most often the
#: robot arm, which is in frame and which the detector has never been trained on
#: - that assumption is violated and the result lands somewhere impossible.
#:
#: Measured on the live cell: four phantom detections on the arm back-projected
#: to x = -0.303, -0.176, -0.160 and -0.143, i.e. behind the robot's own base,
#: while every one of the nine real objects landed in x 0.04..0.45. The
#: separation is not marginal, which is what makes this safe.
#:
#: The table spans link0 x -0.16..0.64; the lower bound here is tighter than
#: that on purpose, because the strip behind the base is occupied by the robot
#: and is not somewhere an object can be placed or reached.
#:
#: This is a guard, not a cure. The detector should not be hallucinating on the
#: arm at all - see tools/synth_dataset/README.md on hard negatives.
WORKSPACE_X = (-0.02, 0.66)
WORKSPACE_Y = (-0.30, 0.78)


def classify_color(image: np.ndarray,
                   corners: List[Tuple[float, float]]) -> str:
    """Modal hue inside a detected box. Pure function of (image, corners).

    Deliberately module-level rather than a method: it is the one piece of
    perception whose correctness can be checked offline against ground truth, and
    tools/synth_dataset/validate.py does exactly that on rendered frames where
    the spawned colour is known. Keeping it a method would have forced the
    validator to reimplement the logic, and a reimplementation agreeing with
    itself proves nothing about what the robot actually runs.

    Masks to the polygon and erodes it, because the outermost ring of an OBB is
    usually half table. Pixels too grey to carry a meaningful hue are dropped
    rather than being allowed to vote.
    """
    h, w = image.shape[:2]
    poly = np.array([[int(round(u)), int(round(v))] for u, v in corners],
                    dtype=np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    # Shrink toward the centre so edge pixels do not vote.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    if cv2.countNonZero(mask) < 12:
        # Eroded away - the box is tiny. Fall back to the un-eroded polygon.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sel = (mask > 0) & (hsv[:, :, 1] >= MIN_SATURATION) & (hsv[:, :, 2] >= MIN_VALUE)

    # Too few saturated pixels means the object has no hue worth reading - which
    # is the correct answer for a bolt, whose DAE is bare grey metal, not a
    # failure to classify.
    hues = hsv[:, :, 0][sel]
    if hues.size < 8:
        return 'grey'

    counts: Dict[str, int] = {}
    for name, lo, hi in HUE_BANDS:
        n = int(np.count_nonzero((hues >= lo) & (hues <= hi)))
        counts[name] = counts.get(name, 0) + n

    best = max(counts, key=counts.get)
    # Require a plurality worth believing, or say so instead of guessing.
    return best if counts[best] >= 0.35 * hues.size else 'unknown'


class Track:
    __slots__ = ('id', 'x', 'y', 'class_name', 'color', 'misses')

    def __init__(self, tid: int, x: float, y: float, class_name: str, color: str):
        self.id = tid
        self.x = x
        self.y = y
        self.class_name = class_name
        self.color = color
        self.misses = 0


class WorldModel(Node):

    def __init__(self):
        super().__init__('world_model')

        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('max_graspable_width', 0.070)
        self.declare_parameter('image_topic', '/image_raw')

        self.max_graspable = float(self.get_parameter('max_graspable_width').value)

        self._image: Optional[np.ndarray] = None
        self._detections: Optional[Yolov8Inference] = None
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        #: Detections dropped for back-projecting outside the workspace. Counted
        #: and reported rather than silently swallowed: a rising number means the
        #: detector is firing on the arm, which is a training-data problem, and
        #: a filter that hides it would just move the bug somewhere harder to see.
        self._rejected = 0
        self._reported_rejected = 0

        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_cb, 10)
        self.create_subscription(
            Yolov8Inference, '/Yolov8_Inference', self._detect_cb, 10)

        self.pub = self.create_publisher(WorldState, '/world_state', 10)
        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(f'world_model publishing /world_state at {rate} Hz')

    def _image_cb(self, msg: Image):
        try:
            self._image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().warning(f'image conversion failed: {exc}')

    def _detect_cb(self, msg: Yolov8Inference):
        self._detections = msg

    # ------------------------------------------------------------------ colour

    def _classify_color(self, corners: List[Tuple[float, float]]) -> str:
        """Colour of the object in this box, or 'unknown' with no image yet."""
        if self._image is None:
            return 'unknown'
        return classify_color(self._image, corners)

    # -------------------------------------------------------------------- ids

    def _assign_id(self, x: float, y: float, class_name: str,
                   claimed: set) -> Tuple[int, bool]:
        """Nearest unclaimed track of the same class, else a new id."""
        best_id, best_d = None, ID_MATCH_RADIUS
        for tid, t in self._tracks.items():
            if tid in claimed or t.class_name != class_name:
                continue
            d = math.hypot(t.x - x, t.y - y)
            if d < best_d:
                best_id, best_d = tid, d
        if best_id is not None:
            return best_id, False
        tid = self._next_id
        self._next_id += 1
        return tid, True

    # ---------------------------------------------------------------- publish

    def _publish(self):
        msg = WorldState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'panda_link0'

        detections = self._detections
        if detections is None:
            self.pub.publish(msg)
            return

        claimed = set()
        seen_now = {}

        for det in detections.yolov8_inference:
            pts = list(det.coordinates)
            if len(pts) != 8:
                continue
            corners = [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]

            plane_z = OBJECT_CENTROID_Z.get(det.class_name, DEFAULT_CENTROID_Z)
            u, v = obb_center(corners)

            # A bolt's box encloses its hex head as well as its shaft, so the
            # box centre is not a point the jaw can close on - measured 34.2 mm
            # off, which put every grasp clear of a 24 mm shaft. Shift toward
            # the narrow end. Returns None when the head end cannot be told
            # apart, and then the centre is used unchanged: a wrong shift is
            # twice as bad as no shift.
            if det.class_name == 'bolt':
                shifted = bolt_grasp_pixel(self._image, corners, plane_z)
                if shifted is not None:
                    u, v = shifted

            x, y = CAMERA.pixel_to_link0(u, v, plane_z)

            # A detection that back-projects off the table cannot be a table
            # object; it is the arm, or a box straddling something it should not.
            # Dropping it here keeps phantom objects out of /world_state entirely,
            # so the planner never sees one and never tries to pick it.
            if not (WORKSPACE_X[0] <= x <= WORKSPACE_X[1]
                    and WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1]):
                self._rejected += 1
                continue

            # Pixel sizes scale to metres by the same depth the position used;
            # at depth d one pixel spans d/fx metres.
            depth = CAMERA.depth_to(plane_z)
            long_px, short_px = obb_axes_px(corners)
            long_m = long_px * depth / CAMERA.fx
            short_m = short_px * depth / CAMERA.fx

            color = self._classify_color(corners)
            tid, is_new = self._assign_id(x, y, det.class_name, claimed)
            claimed.add(tid)

            track = self._tracks.get(tid)
            if track is None:
                track = Track(tid, x, y, det.class_name, color)
                self._tracks[tid] = track
            else:
                track.x, track.y = x, y
                track.misses = 0
                # Colour is a property of the object, not of this frame. Keep
                # the first confident answer so a momentary specular highlight
                # does not rename "the yellow cube" mid-task.
                if track.color == 'unknown' and color != 'unknown':
                    track.color = color
            seen_now[tid] = True

            obj = DetectedObject()
            obj.id = tid
            obj.class_name = det.class_name
            obj.color = track.color
            obj.x = float(x)
            obj.y = float(y)
            obj.yaw = float(obb_long_axis_yaw(corners))
            obj.grasp_yaw = float(grasp_yaw(corners))
            obj.long_axis = float(long_m)
            obj.short_axis = float(short_m)
            obj.graspable = bool(short_m <= self.max_graspable)
            obj.receptacle = False
            obj.confidence = 1.0
            msg.objects.append(obj)

        # Age out tracks that have gone missing for a while.
        for tid in list(self._tracks):
            if tid not in seen_now:
                self._tracks[tid].misses += 1
                if self._tracks[tid].misses > ID_MAX_MISSES:
                    del self._tracks[tid]

        # Surface the rejection count periodically. Silent filtering would make
        # a detector that hallucinates on the arm look like a detector that
        # works, right up until the filter's bounds stop covering the phantoms.
        if self._rejected - self._reported_rejected >= 100:
            self._reported_rejected = self._rejected
            self.get_logger().warning(
                f'{self._rejected} detections dropped for back-projecting '
                f'outside the workspace. Most likely the detector is firing on '
                f'the robot arm - see tools/synth_dataset/README.md on hard '
                f'negatives. The filter is a guard, not the fix.')

        self.pub.publish(msg)


def main():
    rclpy.init(args=None)
    node = WorldModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
