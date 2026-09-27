#!/usr/bin/env python3
"""Save one annotated frame per tool call, for a figure that pairs the
planner's words with what the arm actually did.

    # terminal 1: the cell, then this
    python3 tools/capture_episode.py --out docs/episode

    # terminal 2: run one command, or a benchmark task
    ros2 run llm_task_planner llm_planner --ros-args -p backend:=deepinfra

It tails the newest decision log and, whenever a `tool_call` entry appears,
saves the camera frame as it is at that moment together with the call and its
result. Frames are therefore synchronised to the planner's decisions rather
than to a wall-clock interval, which is what `snapshot.py --watch` gives and
what made the first attempt at this figure unusable: a frame every three
seconds lands mid-motion and belongs to no particular decision.

The frame saved for call N is the state *after* N completed - the tool result
is already in the log when the entry is written - so a panel reads as
"this is what the arm had done by the time it said this".

Writes `<out>/NN_tool.png` plus `<out>/episode.json` holding the pairing, so
the figure script does not have to re-parse the log.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from yolov8_msgs.msg import WorldState, Yolov8Inference
from yolov8_obb.overlay import draw_overlay

RUNS = os.path.expanduser("~/moveit2_obb/runs")


class Grabber(Node):
    def __init__(self):
        super().__init__("capture_episode")
        self.bridge = CvBridge()
        self.image = None
        self.detections: Optional[Yolov8Inference] = None
        self.world: Optional[WorldState] = None
        #: The object the call names, outlined in the frame - the fastest way
        #: to see a grounding error in a figure, per overlay.draw_overlay.
        self.highlight: Optional[int] = None
        self.create_subscription(Image, "/image_raw", self._image, 10)
        self.create_subscription(Yolov8Inference, "/Yolov8_Inference",
                                 lambda m: setattr(self, "detections", m), 10)
        self.create_subscription(WorldState, "/world_state",
                                 lambda m: setattr(self, "world", m), 10)

    def _image(self, msg: Image):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:                                 # noqa: BLE001
            self.get_logger().warning(f"image conversion failed: {exc}")

    def frame(self, label: str):
        if self.image is None:
            return None
        return draw_overlay(self.image.copy(), self.detections, self.world,
                            caption=label, highlight_id=self.highlight)


def newest_log() -> Optional[str]:
    files = glob.glob(os.path.join(RUNS, "*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/episode"))
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--settle", type=float, default=0.6,
                    help="seconds to wait after a call before grabbing, so the "
                         "frame shows the finished motion rather than the arm "
                         "still on its way out of it")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rclpy.init()
    node = Grabber()
    deadline = time.time() + 20.0
    while time.time() < deadline and node.image is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.image is None:
        print("no /image_raw after 20 s - is the cell up?", file=sys.stderr)
        return 1

    path = newest_log()
    if path is None:
        print(f"no decision log in {RUNS}", file=sys.stderr)
        return 1
    print(f"watching {os.path.basename(path)}; run a command now")

    saved, episode = 0, []
    fh = open(path, encoding="utf-8")
    fh.seek(0, os.SEEK_END)
    try:
        while saved < args.max_calls:
            line = fh.readline()
            if not line:
                newer = newest_log()
                if newer and newer != path:      # a new run started
                    fh.close()
                    path, fh = newer, open(newer, encoding="utf-8")
                    print(f"switched to {os.path.basename(newer)}")
                    continue
                rclpy.spin_once(node, timeout_sec=0.1)
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "command":
                print(f'command: {ev.get("command", "")!r}')
                episode.append({"event": "command", "text": ev.get("command", "")})
                continue
            if ev.get("event") != "tool_call":
                continue

            node.highlight = ev.get("input", {}).get("object_id")
            end = time.time() + args.settle
            while time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.05)

            saved += 1
            call = "%s(%s)" % (ev["tool"], json.dumps(ev.get("input", {})))
            head = str(ev.get("result_head", ""))[:90]
            name = "%02d_%s.png" % (saved, ev["tool"])
            img = node.frame("%d. %s" % (saved, call))
            if img is None:
                continue
            cv2.imwrite(os.path.join(args.out, name), img)
            episode.append({"event": "tool_call", "n": saved, "file": name,
                            "tool": ev["tool"], "input": ev.get("input", {}),
                            "is_error": bool(ev.get("is_error")),
                            "result": head})
            print(f"  {saved:2d}  {call[:60]:<60} -> {head[:50]}")
    except KeyboardInterrupt:
        pass
    finally:
        with open(os.path.join(args.out, "episode.json"), "w", encoding="utf-8") as out:
            json.dump(episode, out, indent=2)
        print(f"\n{saved} frames + episode.json in {args.out}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
