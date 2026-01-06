from setuptools import find_packages, setup

package_name = "lerobot_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    zip_safe=True,
    maintainer="Sandro Covo",
    maintainer_email="sandro@sandrocovo.ch",
    description="TODO: Package description",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "policy_controller = lerobot_ros.policy_controller:main",
            "dataset_recorder = lerobot_ros.recorder:main",
            "episode_tracker = lerobot_ros.episode_tracker_node:main",
            "so101_leader = lerobot_ros.so101.leader:main",
            "so101_follower = lerobot_ros.so101.follower:main",
        ],
    },
)
