from typing import Any, Callable, Dict, List

class FrameCollector:
    """A precise frame collector that captures messages at fixed intervals.

    The timing of frame capture is done in Rust for precision.
    Messages are stored as Python objects and sent to Python for processing.
    """

    def __init__(self, topic_names: List[str], fps: float) -> None:
        """Create a new FrameCollector.

        Args:
            topic_names: List of topic names to collect messages for
            fps: Frames per second (how often to capture frames)
        """
        ...

    def add_message(self, topic_name: str, message: Any) -> None:
        """Add a message to the current frame for a specific topic.

        Args:
            topic_name: The name of the topic
            message: The ROS message (as a Python object)
        """
        ...

    def register_callback(
        self, callback: Callable[[Dict[str, List[Any]], float], None]
    ) -> None:
        """Register a callback to be called with each frame.

        The callback should accept two arguments:
            - frame: Dict[str, List[Any]] mapping topic names to lists of messages
            - timestamp: float, the precise timestamp when the frame was captured
        """
        ...

    def stop(self) -> None:
        """Stop the collector threads."""
        ...

    def is_running(self) -> bool:
        """Check if the collector is running."""
        ...
