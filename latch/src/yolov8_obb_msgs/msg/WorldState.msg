# Everything the planner is allowed to know about the scene.
#
# This is the contract between perception and the language layer. The planner
# holds no private knowledge of the robot or the scene - it resolves referring
# expressions ("the leftmost yellow cube") by filtering and sorting this list.
# Keeping it that way is what makes the planning loop swappable.
#
# Carries DETECTED objects only. The bins are static and are never perceived,
# so world_model does not assert them - that would make perception depend on the
# task layer. The planner merges them in from llm_task_planner.scene, which is
# the same source the skill server places into, so the two cannot disagree about
# where a bin is. Entries with receptacle=true therefore only ever appear after
# that merge, never on the wire from world_model.

std_msgs/Header header
DetectedObject[] objects
