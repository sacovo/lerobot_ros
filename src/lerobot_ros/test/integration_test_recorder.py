#!/usr/bin/env python3
import os
import subprocess
import time
import shutil
import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from lerobot_interfaces.srv import NewDataset, StartEpisode, EndEpisode
from std_srvs.srv import Trigger

# Configuration
CONFIG_FILE = "src/lerobot_ros/config/test_recorder.toml"
DATASET_ROOT = "/tmp/lerobot_test_datasets"
DATASET_NAME = "test_dataset"

def main():
    print("Starting Integration Test for Recorder...")

    # 1. Cleanup previous runs
    if os.path.exists(DATASET_ROOT):
        print(f"Cleaning up {DATASET_ROOT}...")
        shutil.rmtree(DATASET_ROOT)

    # 2. Start the recorder node in the background
    # We source the .venv and the install folder as requested.
    # We use a process group to ensure we can kill the node and its children later.
    recorder_cmd = (
        "source .venv/bin/activate && "
        "source install/setup.bash && "
        f"export CONFIG_PATH={CONFIG_FILE} && "
        "ros2 run lerobot_ros dataset_recorder"
    )
    
    print(f"Launching recorder node with command: {recorder_cmd}")
    recorder_process = subprocess.Popen(
        ["bash", "-c", recorder_cmd],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Give some time for the node to start
    time.sleep(5)

    # Initialize ROS2 client
    rclpy.init()
    node = Node("test_client")

    try:
        # 3. Wait for services to become available
        services = {
            "new_dataset": NewDataset,
            "start_episode": StartEpisode,
            "end_episode": EndEpisode,
            "store_episodes": Trigger
        }
        
        for srv_name, srv_type in services.items():
            client = node.create_client(srv_type, srv_name)
            print(f"Waiting for service {srv_name}...")
            if not client.wait_for_service(timeout_sec=20.0):
                raise RuntimeError(f"Service {srv_name} not available")

        # 4. Create a new dataset
        print(f"Creating new dataset: {DATASET_NAME}...")
        client = node.create_client(NewDataset, "new_dataset")
        req = NewDataset.Request()
        req.repo_id = DATASET_NAME
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        result = future.result()
        if not result.success:
            raise RuntimeError(f"Failed to create dataset: {result.msg}")
        print("Dataset created successfully.")

        # 5. Start a few episodes
        for ep in range(1, 3):
            print(f"\n--- Episode {ep} ---")
            print(f"Starting episode {ep}...")
            client = node.create_client(StartEpisode, "start_episode")
            req = StartEpisode.Request()
            req.task = f"test_task_{ep}"
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            episode_id = future.result().episode_id
            print(f"Episode {episode_id} started.")

            # 6. Publish data on the topics
            print(f"Publishing data for episode {ep}...")
            pub = node.create_publisher(Float32, "/test/float", 10)
            for i in range(20):
                msg = Float32()
                msg.data = float(i + ep * 100)
                pub.publish(msg)
                time.sleep(0.1) # Publish at 10Hz for 2s
            
            # 7. End episode
            print(f"Ending episode {ep}...")
            client = node.create_client(EndEpisode, "end_episode")
            req = EndEpisode.Request()
            req.discard = False
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
            print(f"Episode ended with {future.result().frames} frames.")

        # 8. Store episodes
        print("\nStoring episodes...")
        client = node.create_client(Trigger, "store_episodes")
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        if not future.result().success:
            raise RuntimeError(f"Failed to store episodes: {future.result().message}")
        print("Episodes storage triggered.")

        # Wait for background storage thread in recorder.py to finish
        print("Waiting for storage to complete (approx 10s)...")
        time.sleep(10)

        # 9. Verify the episodes have been created
        dataset_path = os.path.join(DATASET_ROOT, DATASET_NAME)
        if not os.path.exists(dataset_path):
            raise RuntimeError(f"Dataset directory not found at {dataset_path}")
        
        # Check for expected LeRobot dataset structure
        print(f"Verifying dataset at {dataset_path}...")
        files = os.listdir(dataset_path)
        print(f"Files found: {files}")
        
        # LeRobotDataset usually has a 'meta' directory or 'metadata.json' and 'data' directory
        if not files:
            raise RuntimeError("Dataset directory is empty")

        print("\nTest PASSED successfully!")

    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        # Capture some output from the recorder process if possible
        stdout, stderr = recorder_process.communicate(timeout=1)
        print("--- Recorder STDOUT ---")
        print(stdout)
        print("--- Recorder STDERR ---")
        print(stderr)
        raise
    finally:
        # Cleanup ROS2
        print("\nCleaning up ROS2...")
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

        # Kill the recorder node
        if recorder_process:
            print("Killing recorder node...")
            try:
                os.killpg(os.getpgid(recorder_process.pid), signal.SIGTERM)
                recorder_process.wait(timeout=5)
            except Exception as e:
                print(f"Error killing process: {e}")

        # 10. Delete the dataset again
        if os.path.exists(DATASET_ROOT):
            print(f"Deleting test dataset at {DATASET_ROOT}...")
            shutil.rmtree(DATASET_ROOT)

if __name__ == "__main__":
    main()
