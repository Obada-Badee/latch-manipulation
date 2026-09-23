"""Task layer: manipulation skills and the language planner that drives them.

Deliberately split from panda_moveit_config (which describes the robot) and from
yolov8_obb (which describes what the camera sees). The boundary that matters:

    llm_planner   decides WHAT to do      - knows only /world_state and the skills
    skill_server  decides HOW to do it    - knows the robot, not the task
    scene.py      constants neither can perceive (bin poses, gripper geometry)
    grasp.py      orientation maths

Keeping llm_planner ignorant of the robot is what makes the planning loop
replaceable without touching anything else.
"""
