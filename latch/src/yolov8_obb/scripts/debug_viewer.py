#!/usr/bin/env python3
"""Live view of what the robot sees and what it calls things.

    ros2 run yolov8_obb debug_viewer.py

Replaces UI/bolt_selector.py, which was the system's original entry point:
click a bolt, and a hardcoded pick-and-place macro ran. Natural language is the
input now, so a clicking GUI is no longer a control surface - but watching the
detector's boxes and the planner's ids appear on the frame is still the fastest
way to see a grounding error for what it is, so the view is worth keeping.

WHAT WAS WRONG WITH THE OLD ONE, since "rewritten" deserves reasons:

  * Three separate rclpy Nodes in one process (image, yolo, publisher), each
    spun by hand from a Qt timer.
  * `rclpy.spin_once(node)` with no timeout, called from that timer. If the
    image stream stalls, the call blocks forever and the whole GUI freezes -
    including the window that would have told you the stream had stalled.
  * Its own copy of the camera intrinsics and back-projection, including a
    depth plane of `z = 0.7`. That is the camera's height above the link0
    plane, not above the objects, and it had already been corrected to a
    per-class centroid plane everywhere else. The viewer was quietly showing
    positions the rest of the system disagreed with.
  * It published `/target_point`, which nothing has subscribed to since
    arm_control_from_UI.py was replaced by skill_server.

This version is one node, uses a bounded spin, and imports the shared
projection and overlay modules rather than reimplementing them - so it cannot
disagree with the world model about where anything is. It publishes nothing:
a debug view that can move the arm is not a debug view.

Press q to quit, s to save a still.
"""

import time
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from yolov8_msgs.msg import WorldState, Yolov8Inference

from yolov8_obb.overlay import draw_overlay

WINDOW = 'moveit2_obb - what the robot sees'

#: How long a topic may go quiet before the caption says so. Long enough to
#: ride out a slow detector frame, short enough to notice a dead node.
STALE_S = 3.0


class DebugViewer(Node):

    def __init__(self):
        super().__init__('debug_viewer')
        self.declare_parameter('save_dir', '')

        self.bridge = CvBridge()
        self.image = None
        self.detections: Optional[Yolov8Inference] = None
        self.world: Optional[WorldState] = None
        self.seen = {'image': 0.0, 'detections': 0.0, 'world': 0.0}

        self.create_subscription(Image, '/image_raw', self._on_image, 10)
        self.create_subscription(Yolov8Inference, '/Yolov8_Inference',
                                 self._on_detections, 10)
        self.create_subscription(WorldState, '/world_state', self._on_world, 10)
        self.get_logger().info('debug_viewer: q to quit, s to save a still')

    def _on_image(self, msg: Image):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.seen['image'] = time.monotonic()
        except Exception as exc:                                # noqa: BLE001
            self.get_logger().warning(f'image conversion failed: {exc}')

    def _on_detections(self, msg: Yolov8Inference):
        self.detections = msg
        self.seen['detections'] = time.monotonic()

    def _on_world(self, msg: WorldState):
        self.world = msg
        self.seen['world'] = time.monotonic()

    def stale(self):
        """Which topics have gone quiet. Reported rather than left to be
        inferred from a frozen picture - the old viewer's failure mode was
        looking exactly like a working one."""
        now = time.monotonic()
        return [name for name, at in self.seen.items()
                if at == 0.0 or now - at > STALE_S]


def main():
    rclpy.init(args=None)
    node = DebugViewer()
    save_dir = str(node.get_parameter('save_dir').value) or '.'
    saved = 0

    try:
        while rclpy.ok():
            # Bounded, always. This is the bug that froze the old GUI: an
            # unbounded spin_once in a display loop hands control to a stream
            # that may never speak again.
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.image is None:
                continue

            stale = node.stale()
            frame = draw_overlay(
                node.image, node.detections, node.world,
                caption='live',
                extra_lines=([f'STALE: {", ".join(stale)}'] if stale else []))
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                path = f'{save_dir}/debug_view_{saved:03d}.png'
                cv2.imwrite(path, frame)
                node.get_logger().info(f'saved {path}')
                saved += 1
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
