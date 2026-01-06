import sys
import termios
import threading
import traceback
import tty
from typing import cast

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from lerobot_interfaces.srv import EndEpisode, NewDataset, StartEpisode


def call_service(node: Node, service_name: str, request, srv_type):
    client = node.create_client(srv_type, service_name)
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"Service {service_name} not available")
    result = client.call(request)
    return result


def getch():
    """Get a single character from stdin without pressing enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class RecorderCLI:
    def __init__(self):
        rclpy.init(args=sys.argv)
        self.node = Node("recorder_cli")

        self.tasks = list(
            self.node.declare_parameter(
                "tasks",
                [
                    "toggle main switch",
                    "toggle lever switch",
                    "turn switch",
                    "pickup plug",
                    "insert plug",
                    "rotate power button",
                    "pickup plate",
                    "place plate",
                ],
            )
            .get_parameter_value()
            .string_array_value
        )

        # State
        self.dataset_created = False
        self.episode_running = False
        self.current_episode_id = None
        self.current_task = None
        self.frame_count = 0
        self.episode_start_frame_count = 0

        # Start ROS thread
        self.ros_thread = threading.Thread(
            target=lambda: rclpy.spin(self.node), daemon=True
        )
        self.ros_thread.start()

    def clear_screen(self):
        print("\033[2J\033[H", end="")

    def display_status(self):
        self.clear_screen()
        print("=" * 60)
        print("ROS2 RECORDER CLI")
        print("=" * 60)

        if self.episode_running:
            print(f"EPISODE RUNNING: {self.current_episode_id}")
            print(f"Task: {self.current_task}")

        print("-" * 60)

        if not self.dataset_created:
            print("Status: No dataset created")
        elif self.episode_running:
            print("Status: Episode running")
        else:
            print("Status: Dataset ready")

        print("-" * 60)

    def show_menu(self):
        if not self.dataset_created:
            print("\nPlease create a dataset first")
            return

        if self.episode_running:
            print("\nEpisode Controls:")
            print("  [S] Save episode")
            print("  [D] Discard episode")
        else:
            print("\nSelect task to start episode:")
            for i, task in enumerate(self.tasks, 1):
                print(f"  [{i}] {task.replace('_', ' ').title()}")
            print("\nOther Options:")
            print("  [S] Store episodes")

        print("  [Q] Quit")
        print("\nPress a key...")

    def create_dataset(self):
        print("\nEnter dataset repo ID: ", end="", flush=True)
        repo_id = input().strip()

        if not repo_id:
            print("Error: Empty repo ID")
            input("Press Enter to continue...")
            return False

        try:
            request = NewDataset.Request()
            request.repo_id = repo_id
            result = call_service(self.node, "/new_dataset", request, NewDataset)
            result = cast(NewDataset.Response, result)

            if result.success:
                self.dataset_created = True
                print("✓ Dataset created successfully!")
                input("Press Enter to continue...")
                return True
            else:
                print(f"✗ Error: {result.msg}")
                input("Press Enter to continue...")
                return False
        except Exception as e:
            print(f"✗ Failed to create dataset: {e}")
            input("Press Enter to continue...")
            return False

    def start_episode(self, task_index):
        if task_index < 1 or task_index > len(self.tasks):
            return

        task = self.tasks[task_index - 1]

        try:
            request = StartEpisode.Request()
            request.task = task
            result = call_service(self.node, "/start_episode", request, StartEpisode)
            result = cast(StartEpisode.Response, result)

            self.episode_running = True
            self.current_episode_id = result.episode_id
            self.current_task = task

            print(f"✓ Started episode {result.episode_id} for task: {task}")
        except Exception as e:
            print(f"✗ Failed to start episode: {e}")
            input("Press Enter to continue...")

    def end_episode(self, discard=False):
        if not self.episode_running:
            return

        try:
            request = EndEpisode.Request()
            request.discard = discard
            result = call_service(self.node, "/end_episode", request, EndEpisode)
            result = cast(EndEpisode.Response, result)

            episode_frames = result.frames
            action = "discarded" if discard else "saved"

            print(f"✓ Episode {action}")
            print(f"  Episode frames: {episode_frames}")
            print(f"  Service reported: {result.frames}")

            self.episode_running = False
            self.current_episode_id = None
            self.current_task = None

        except Exception as e:
            print(f"✗ Failed to end episode: {e}")
            input("Press Enter to continue...")

    def store_episodes(self):
        try:
            print("\nStoring episodes...")
            request = Trigger.Request()
            result = call_service(self.node, "/store_episodes", request, Trigger)
            print(f"✓ Stored episodes: {result.success}")
        except Exception as e:
            print(f"✗ Failed to store episodes: {e}")
            input("Press Enter to continue...")

    def run(self):
        print("Starting ROS2 Recorder CLI...")
        print("Waiting for ROS services...")

        # Create dataset first
        while not self.dataset_created:
            self.display_status()
            if not self.create_dataset():
                continue

        # Main loop
        while True:
            self.display_status()
            self.show_menu()

            try:
                key = getch().lower()

                if key == "q":
                    print("\nExiting...")
                    break

                elif not self.episode_running:
                    # Task selection mode
                    if key.isdigit():
                        task_num = int(key)
                        if 1 <= task_num <= len(self.tasks):
                            self.start_episode(task_num)
                    elif key == "s":
                        self.store_episodes()

                else:
                    # Episode control mode
                    if key == "s":
                        self.end_episode(discard=False)
                    elif key == "d":
                        self.end_episode(discard=True)

            except KeyboardInterrupt:
                print("\nExiting...")
                if self.episode_running:
                    self.end_episode(discard=True)
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")

        # Cleanup
        try:
            self.node.destroy_node()
            rclpy.shutdown()
        except:
            pass


def main():
    try:
        cli = RecorderCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
