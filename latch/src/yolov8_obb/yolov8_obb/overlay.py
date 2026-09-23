"""Drawing what the robot sees onto the frame it saw it in.

Shared by tools/snapshot.py (save a still) and scripts/debug_viewer.py (watch
it live), so the two cannot drift. That mattering is not hypothetical: the
previous viewer, UI/bolt_selector.py, carried its own copy of the camera
intrinsics and its own back-projection, including a depth plane of z = 0.7 that
had already been corrected everywhere else. A viewer that disagrees with the
system it is meant to be showing you is worse than no viewer.

Nothing here computes anything. Positions come from /world_state, boxes come
from /Yolov8_Inference, and the only arithmetic is re-projecting a grounded
position back into pixels using yolov8_obb.projection - the same module the
world model used to produce it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from yolov8_obb.projection import (CAMERA, DEFAULT_CENTROID_Z,
                                   OBJECT_CENTROID_Z)

#: BGR, per class, so a misclassification shows up without reading the text.
CLASS_COLORS = {
    'cube': (80, 220, 80),
    'cylinder': (240, 180, 60),
    'bolt': (80, 160, 250),
}
DEFAULT_COLOR = (200, 200, 200)
HIGHLIGHT = (60, 60, 255)

PANEL_BG = (32, 32, 32)
TEXT = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _shadowed(canvas, text, org, colour, scale=0.4):
    """Text with a dark outline. The table is light and the shadow under it is
    dark, and a label has to stay readable across both."""
    cv2.putText(canvas, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, FONT, scale, colour, 1, cv2.LINE_AA)


def draw_overlay(image: np.ndarray, detections=None, world=None,
                 caption: str = '', highlight_id: Optional[int] = None,
                 extra_lines: Sequence[str] = ()) -> np.ndarray:
    """Return a new frame with detections, grounded ids and a caption strip.

    `highlight_id` outlines one world-state object - use it to show which
    object the planner just chose, which is the fastest way to see a grounding
    error for what it is.
    """
    canvas = image.copy()

    # Detector boxes first, so id labels sit on top of them.
    for det in (detections.yolov8_inference if detections else []):
        pts = list(det.coordinates)
        if len(pts) != 8:
            continue
        poly = np.array([[int(pts[i]), int(pts[i + 1])] for i in range(0, 8, 2)],
                        dtype=np.int32)
        cv2.polylines(canvas, [poly],
                      True, CLASS_COLORS.get(det.class_name, DEFAULT_COLOR),
                      2, cv2.LINE_AA)

    # Grounded objects, drawn at their back-projected position re-projected
    # into the image. Deliberately NOT at the box centre: when the marker and
    # the box disagree, the back-projection is wrong, and that is exactly the
    # kind of thing a debug view exists to make visible.
    for obj in (world.objects if world else []):
        plane_z = OBJECT_CENTROID_Z.get(obj.class_name, DEFAULT_CENTROID_Z)
        u, v = CAMERA.link0_to_pixel(obj.x, obj.y, plane_z)
        u, v = int(round(u)), int(round(v))
        colour = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)
        if highlight_id is not None and obj.id == highlight_id:
            cv2.circle(canvas, (u, v), 16, HIGHLIGHT, 2, cv2.LINE_AA)
            colour = HIGHLIGHT
        cv2.circle(canvas, (u, v), 3, colour, -1, cv2.LINE_AA)
        _shadowed(canvas, f'{obj.id} {obj.color} {obj.class_name}',
                  (u + 7, v - 7), colour)

    n_det = len(detections.yolov8_inference) if detections else 0
    n_obj = len(world.objects) if world else 0
    lines = ([caption] if caption else []) + [
        f'{n_det} detections, {n_obj} grounded objects'
        + ('' if detections else '   [detector not publishing]')
        + ('' if world else '   [world_model not publishing]')
    ] + list(extra_lines)

    strip = 18 * len(lines) + 10
    out = np.full((canvas.shape[0] + strip, canvas.shape[1], 3), PANEL_BG,
                  dtype=np.uint8)
    out[:canvas.shape[0]] = canvas
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, canvas.shape[0] + 18 + i * 18), FONT, 0.45,
                    TEXT, 1, cv2.LINE_AA)
    return out
