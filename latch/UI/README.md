# UI/ — superseded

`bolt_selector.py` was this project's original entry point: a Qt window showing
the camera feed with the detector's boxes drawn on it, where clicking a bolt
published a target and ran a hardcoded pick-and-place macro.

**It is no longer the way the system is driven, and it should not be run.** The
input is natural language now (`ros2 run llm_task_planner llm_planner`), and the
`/target_point` topic it publishes to has had no subscriber since
`arm_control_from_UI.py` was replaced by `skill_server`.

The viewing part was worth keeping — watching the boxes and ids appear on the
frame is still the quickest way to see a grounding error for what it is — so it
was rewritten as a plain ROS node:

```bash
ros2 run yolov8_obb debug_viewer.py     # live, q to quit, s to save a still
python3 tools/snapshot.py out.png       # one annotated still
```

Both draw through `yolov8_obb.overlay` and position through
`yolov8_obb.projection`, so neither can disagree with the world model about
where anything is.

## Why it was not simply updated

Four defects, and the third is the one that matters:

1. **Three `rclpy.Node` objects in one process** — one for the image, one for
   detections, one to publish — each spun by hand from a Qt timer.
2. **`rclpy.spin_once(node)` with no timeout**, called from that timer. If the
   image stream stalls, that call blocks forever and the whole GUI freezes,
   including the window that would have told you the stream had stalled.
3. **Its own copy of the camera intrinsics and back-projection**, including a
   depth plane of `z = 0.7`. That is the camera's height above the *link0
   plane*, not above the objects, and it had already been corrected to a
   per-class centroid plane everywhere else in the system
   (`yolov8_obb/projection.py`). The viewer was quietly showing positions the
   rest of the system disagreed with — the worst thing a debug tool can do.
4. **It could move the arm.** A view that can drive the robot is not a view.

The files are left here rather than deleted because this workspace is not under
version control; deleting them would be unrecoverable. Treat them as a record of
where the project started.
