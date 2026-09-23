# One object the robot knows about.
#
# Everything here is expressed in the panda_link0 frame: +X away from the robot,
# +Y to the robot's LEFT, +Z up. "left" and "right" in a user command are
# resolved against this frame and nothing else - camera-left, robot-left and
# viewer-left are three different directions, and picking one and stating it is
# the only way spatial referring expressions stay meaningful.
#
# Produced by world_model from /Yolov8_Inference; consumed by llm_planner, which
# sees this as the entire world.

int32 id                # stable across frames while the object stays visible
string class_name       # cube | bolt | cylinder | bin
string color            # red | yellow | blue | green | unknown
float64 x               # metres, panda_link0
float64 y               # metres, panda_link0
float64 yaw             # long-axis orientation, radians, wrapped to [-pi/2, pi/2)

# --- affordances -----------------------------------------------------------
# The OBB's orientation is a genuine grasp affordance rather than a lookup:
# a parallel jaw has to close across the object's SHORT axis, so the hand yaw
# is perpendicular to the long axis above.
float64 grasp_yaw       # radians, what the gripper should adopt to grasp this
float64 long_axis       # metres, OBB long side
float64 short_axis      # metres, OBB short side - must fit the 60 mm jaw
bool graspable          # false for bins, and for anything too wide for the jaw
bool receptacle         # true for bins: a valid `place` destination

float64 confidence      # detector confidence, 0..1
