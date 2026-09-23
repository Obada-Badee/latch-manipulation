"""Bring up the simulated cell: Gazebo, the robot, MoveIt, perception, skills.

Changes from the original:

  * The three controller spawners no longer race the simulator. They used to be
    bare ExecuteProcess actions that fired the instant launch started, before
    gz_ros2_control had loaded its plugin - which worked most of the time and
    failed the rest, looking like a flaky simulator. RegisterEventHandler and
    OnProcessExit were already imported in this file and never used; the
    sequencing was clearly intended and got dropped. They are now chained off
    the spawn, and off each other.

  * arm_control_from_UI.py is replaced by skill_server, which exposes pick /
    place / home as actions instead of one blocking click-driven macro.

  * world_model is started here too, so /world_state exists as soon as the
    detector does.

The LLM planner is deliberately NOT launched here - see llm_planner.launch.py.
It needs to be restartable without restarting the simulator, which is most of
the debugging loop.
"""

import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    robot_description_path = get_package_share_directory('robot_description')
    arm_robot_sim_path = get_package_share_directory('panda_moveit_config')

    gazebo_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            os.path.join(arm_robot_sim_path, 'worlds'), ':' +
            str(Path(robot_description_path).parent.resolve())
        ]
    )

    declared = [
        DeclareLaunchArgument('world', default_value='arm_on_the_table',
                              description='Gz sim world name'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Also start RViz'),
        # Headless runs the server only (`gz sim -s`), with no GUI window. The
        # camera still renders - the ogre2 sensor pipeline does not need a
        # window, which is what tools/wsl/40-check-render.sh established. Useful
        # for automated verification, and for not putting a Gazebo window on
        # someone's desktop unasked.
        DeclareLaunchArgument('verify', default_value='true',
            description='false = skills report success on motion alone (naive baseline ONLY)'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run gz without the GUI'),
        DeclareLaunchArgument('perception', default_value='true',
                              description='Start the detector and world_model'),
        DeclareLaunchArgument('detector_conf', default_value='0.50',
                              description='YOLO confidence threshold. The original '
                                          'used 0.90, tuned for a single easy '
                                          'class; three classes need lower.'),
        DeclareLaunchArgument('weights', default_value='',
                              description='Override the detector checkpoint. Empty '
                                          'means the installed scripts/best.pt.'),
        # dartsim, NOT the project's original bullet-featherstone.
        #
        # bullet-featherstone will not drive the prismatic finger joint with its
        # mimic constraint. Measured on the same scene, same controllers, same
        # commands - only the engine changed:
        #
        #     engine                 open        close       goal reached?
        #     bullet-featherstone    0.6 mm/s    0.2 mm/s    no, ever
        #     dartsim              103.8 mm/s   57.3 mm/s    yes, in <1 s
        #
        # Under bullet-featherstone a 40 mm grasp needed over a minute of jaw
        # travel and never completed, so no grasp could succeed. This is a
        # pre-existing property of the original configuration, not something the
        # LLM/skill layer introduced. Kept as an argument so it can be A/B'd.
        DeclareLaunchArgument('physics_engine',
                              default_value='gz-physics-dartsim-plugin',
                              description='gz physics engine plugin name'),
    ]

    # PythonExpression rather than two IncludeLaunchDescriptions: gz_args is a
    # single string, so the -s flag has to be substituted into it.
    headless_flag = PythonExpression([
        "' -s ' if '", LaunchConfiguration('headless'),
        "'.lower() in ('true', '1') else ''"])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch'),
            '/gz_sim.launch.py']),
        launch_arguments=[
            ('gz_args', [LaunchConfiguration('world'),
                         '.sdf',
                         ' -v 4',
                         ' -r',
                         headless_flag,
                         ' --physics-engine ', LaunchConfiguration('physics_engine')]
             )
        ]
    )

    xacro_file = os.path.join(arm_robot_sim_path, 'config', 'panda.urdf.xacro')
    robot_desc = xacro.process_file(
        xacro_file, mappings={'use_sim': 'true'}).toprettyxml(indent='  ')

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-string', robot_desc,
                   '-x', '0.05', '-y', '0.0', '-z', '1.02',
                   '-R', '0.0', '-P', '0.0', '-Y', '0.0',
                   '-name', 'arm_robot',
                   '-allow_renaming', 'false'],
    )

    moveit_config = (
        MoveItConfigsBuilder(robot_name='panda')
        .robot_description(file_path='config/panda.urdf.xacro')
        .robot_description_semantic(file_path='config/panda.srdf')
        .trajectory_execution(file_path='config/gripper_moveit_controllers.yaml')
        .moveit_cpp(arm_robot_sim_path + '/config/controller_setting.yaml')
        .to_moveit_configs()
    )

    # --- controller spawners, sequenced -------------------------------------
    # joint_state_broadcaster first: the other two are useless without joint
    # feedback, and MoveIt will not plan until it sees a joint state.
    def spawner(name):
        return ExecuteProcess(
            cmd=[f'ros2 run controller_manager spawner {name}'],
            shell=True, output='log')

    jsb = spawner('joint_state_broadcaster')
    arm_ctrl = spawner('panda_arm_controller')
    hand_ctrl = spawner('panda_hand_controller')

    # Wait for the robot to actually exist in Gazebo before asking its
    # controller_manager for anything.
    after_spawn = RegisterEventHandler(
        OnProcessExit(target_action=gz_spawn_entity, on_exit=[jsb]))
    after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=jsb, on_exit=[arm_ctrl]))
    after_arm = RegisterEventHandler(
        OnProcessExit(target_action=arm_ctrl, on_exit=[hand_ctrl]))

    # --- nodes ---------------------------------------------------------------
    skill_server = Node(
        name='skill_server',
        package='llm_task_planner',
        executable='skill_server',
        output='both',
        parameters=[moveit_config.to_dict(), {'use_sim_time': True,
                     'verify': LaunchConfiguration('verify')}],
        # It segfaulted (exit -11) 44 trials into a 75-trial run, under MoveIt
        # after a preempted trajectory - a crash in C++ this package cannot
        # fix. Without respawn the 26 trials after it ran against a dead robot.
        # With it the crash costs the one trial it interrupted: the benchmark
        # waits for the servers to come back, voids that trial, and continues.
        respawn=True,
        respawn_delay=3.0,
    )

    # The detector and world_model are one unit: world_model consumes
    # /Yolov8_Inference and has nothing to publish without it. They used to be
    # split across two launch files, so `perception:=true` started world_model
    # alone and it emitted empty world states indefinitely - which looks exactly
    # like "the detector sees nothing" rather than "the detector is not running".
    detector = Node(
        package='yolov8_obb',
        executable='yolov8_obb_publisher.py',
        name='yolov8_obb_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('perception')),
        parameters=[{
            'use_sim_time': True,
            'conf': LaunchConfiguration('detector_conf'),
            # An empty string means "use the installed default"; the node
            # resolves that itself from its share directory.
            'weights': LaunchConfiguration('weights'),
        }],
    )

    world_model = Node(
        package='yolov8_obb',
        executable='world_model.py',
        name='world_model',
        output='screen',
        condition=IfCondition(LaunchConfiguration('perception')),
        parameters=[{'use_sim_time': True}],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='log',
        parameters=[moveit_config.robot_description, {'use_sim_time': True}],
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=['--frame-id', 'world', '--child-frame-id', 'panda_link0'],
        parameters=[{'use_sim_time': True}],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            # /clock, gz -> ROS only (the '[' direction). Every node in this
            # launch sets use_sim_time:=True, but nothing was carrying Gazebo's
            # clock across, so ROS time sat frozen at 0.000 forever.
            #
            # The consequence is narrow and therefore very easy to misread: a
            # frozen clock breaks *timers* and nothing else. The detector fires
            # on image callbacks and skill_server on action goals, so both
            # worked perfectly - while world_model, whose only output comes from
            # a 5 Hz timer, published nothing at all. /world_state had a
            # publisher, a subscriber, no errors in its log, and a startup
            # message claiming 5 Hz. It looked like a detector problem.
            #
            # Measured before the fix: a node with use_sim_time advanced
            # 0.000 s over 3 s of wall time.
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(arm_robot_sim_path, 'config',
                                      'motion_planning.rviz')],
        parameters=[moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    {'use_sim_time': True}],
    )

    return LaunchDescription(
        declared + [
            gazebo_resource_path,
            gazebo,
            gz_spawn_entity,
            after_spawn,
            after_jsb,
            after_arm,
            robot_state_publisher,
            static_tf,
            bridge,
            skill_server,
            detector,
            world_model,
            rviz_node,
        ]
    )
