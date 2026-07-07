from setuptools import find_packages, setup

package_name = "lerobot_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={
        "lerobot_ros": ["gui/*"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    zip_safe=False,
    maintainer="Sandro Covo",
    maintainer_email="sandro@sandrocovo.ch",
    description="ROS 2 bridge for LeRobot policies: policy controller, dataset recording, bag-to-dataset conversion, the annotation web UI, and TensorRT engine export/packaging.",
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
            "train_episode_tracker = lerobot_ros.episode_tracker:main",
            "so101_leader = lerobot_ros.so101.leader:main",
            "so101_follower = lerobot_ros.so101.follower:main",
            "replay = lerobot_ros.replay:main",
            "bag_to_dataset = lerobot_ros.bag_to_dataset:main",
            "annotation_gui = lerobot_ros.annotation_server:main",
            "init_benchmark_policy = lerobot_ros.init_benchmark_policy:main",
            "convert_policy = lerobot_ros.convert_policy:main",
            "package_engines = lerobot_ros.package_engines:main",
        ],
    },
)
