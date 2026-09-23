from setuptools import find_packages, setup

package_name = 'llm_task_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/llm_planner.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Obada',
    maintainer_email='obadabadee3@hotmail.com',
    description='LLM-driven closed-loop task planning on a Panda arm.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # Skills. Needs the MoveIt config passed as parameters, so it is
            # launched from a launch file rather than run bare.
            'skill_server = llm_task_planner.skill_server:main',
            # The language layer. Reads ANTHROPIC_API_KEY from the environment;
            # restartable without restarting the sim.
            'llm_planner = llm_task_planner.llm_planner:main',
        ],
    },
)
