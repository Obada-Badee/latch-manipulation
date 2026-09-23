"""Start the language planner on its own.

Deliberately separate from moveit_gazebo_obb.py: the planner is the piece that
gets edited and restarted constantly during development, and restarting it must
not mean waiting for Gazebo, MoveIt, and the controllers to come back up.

    ros2 launch llm_task_planner llm_planner.launch.py
    ros2 launch llm_task_planner llm_planner.launch.py dry_run:=true
    ros2 launch llm_task_planner llm_planner.launch.py backend:=groq

Requires ANTHROPIC_API_KEY or GROQ_API_KEY in the environment, depending on the
backend. The planner reads the key there and nowhere else; do not put one in a
parameter file that could be committed. The predecessor project is the reason
that warning is here rather than implied - its key is still retrievable from its
public git history, and no amount of rewriting takes that back.

Note this launches an interactive prompt, so run it in its own terminal - or run
the node directly with `ros2 run llm_task_planner llm_planner`, which is often
easier for a stdin-driven node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'dry_run', default_value='false',
            description='Print the chosen actions without moving the arm'),
        DeclareLaunchArgument(
            'backend', default_value='anthropic',
            description='LLM provider: anthropic | groq'),
        DeclareLaunchArgument(
            'model', default_value='',
            description="Model id. Empty uses the backend's default "
                        "(claude-opus-5 / openai/gpt-oss-120b). To reproduce the "
                        "predecessor project's setup: backend:=groq "
                        "model:=llama-3.3-70b-versatile"),
        DeclareLaunchArgument(
            'effort', default_value='high',
            description='Anthropic only. low | medium | high | xhigh | max'),
        DeclareLaunchArgument(
            'temperature', default_value='0.0',
            description='Groq only. 0 keeps object choice deterministic.'),
        DeclareLaunchArgument(
            'forget', default_value='false',
            description='Clear the robot\'s memory of objects and past commands '
                        'before starting. Memory persists across restarts on '
                        'purpose, so that "put it back" survives a crash - but '
                        'it must NOT persist across a world reset, or the robot '
                        'will report objects into bins that were emptied while '
                        'it was not running. Set this whenever the world is '
                        'reloaded or respawned.'),
        DeclareLaunchArgument(
            'memory_file', default_value='',
            description='Where beliefs and the command log are kept. Empty puts '
                        'it beside the decision logs in ~/moveit2_obb/runs.'),

        Node(
            package='llm_task_planner',
            executable='llm_planner',
            name='llm_planner',
            output='screen',
            emulate_tty=True,      # keeps the interactive prompt readable
            parameters=[{
                'dry_run': LaunchConfiguration('dry_run'),
                'backend': LaunchConfiguration('backend'),
                'model': LaunchConfiguration('model'),
                'effort': LaunchConfiguration('effort'),
                'temperature': LaunchConfiguration('temperature'),
                'forget': LaunchConfiguration('forget'),
                'memory_file': LaunchConfiguration('memory_file'),
                'use_sim_time': True,
            }],
        ),
    ])
