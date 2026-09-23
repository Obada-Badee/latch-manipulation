#!/usr/bin/env python3
"""Save an annotated view of what the robot is seeing, right now.

The overhead camera frame on its own is only half the story - it shows the
scene but not the robot's interpretation of it. This overlays the detector's
oriented boxes and the world model's grounded ids on top, so one image answers
"what is on the table", "what did the detector find", and "what does the
planner call each thing" together. That makes it useful for a write-up and for
debugging a grounding error, which is the same picture either way.

    python3 tools/snapshot.py docs/screenshots/01-before.png
    python3 tools/snapshot.py --label "after: red cube in the red bin" out.png
    python3 tools/snapshot.py --watch 8 --every 3 docs/screenshots/run

`--watch N` saves N frames `--every` seconds, which is how you catch the arm
mid-motion without trying to time a single call by hand.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from yolov8_msgs.msg import WorldState, Yolov8Inference

# Drawing lives in the yolov8_obb package, shared with the live debug viewer,
# so a still and the live view can never disagree about what they are showing.
from yolov8_obb.overlay import draw_overlay


class Snapshotter(Node):

    def __init__(self):
        super().__init__('snapshot')
        self.bridge = CvBridge()
        self.image: Optional[np.ndarray] = None
        self.detections: Optional[Yolov8Inference] = None
        self.world: Optional[WorldState] = None
        self.create_subscription(Image, '/image_raw', self._image, 10)
        self.create_subscription(Yolov8Inference, '/Yolov8_Inference',
                                 lambda m: setattr(self, 'detections', m), 10)
        self.create_subscription(WorldState, '/world_state',
                                 lambda m: setattr(self, 'world', m), 10)

    def _image(self, msg: Image):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:                                # noqa: BLE001
            self.get_logger().warning(f'image conversion failed: {exc}')

    def wait(self, timeout: float = 20.0) -> bool:
        """Wait for a frame. Detections and world state are optional - a
        snapshot of a cell whose detector is down is still worth having, and
        is in fact exactly the picture you want in that situation."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.image is not None:
                return True
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', help='PNG path, or a prefix when --watch is used')
    ap.add_argument('--label', default='', help='caption written into the image')
    ap.add_argument('--watch', type=int, default=0,
                    help='save this many frames instead of one')
    ap.add_argument('--every', type=float, default=3.0,
                    help='seconds between frames when watching')
    args = ap.parse_args(argv)

    rclpy.init()
    node = Snapshotter()
    try:
        if not node.wait():
            print('no camera frame arrived on /image_raw; is the cell up?',
                  file=sys.stderr)
            return 1

        shots = max(1, args.watch)
        for i in range(shots):
            if i:
                deadline = time.time() + args.every
                while time.time() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
            # Drain to the newest frame rather than whatever is latched.
            for _ in range(6):
                rclpy.spin_once(node, timeout_sec=0.05)

            path = (args.out if args.watch == 0
                    else f'{args.out}-{i:02d}.png')
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            caption = args.label if args.watch == 0 else (
                f'{args.label} [+{i * args.every:.0f}s]' if args.label
                else f'+{i * args.every:.0f}s')
            frame = draw_overlay(node.image, node.detections, node.world,
                                 caption)
            cv2.imwrite(path, frame)
            n = len(node.world.objects) if node.world else 0
            print(f'  saved {path}  ({n} grounded objects)')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
