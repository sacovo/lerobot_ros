"""Latching software e-stop, shared by every node that commands the manipulator
autonomously.

See SPEC_ESTOP.md at the repo root for the full contract. Summary: `/e_stop`
(any message) latches a trip; `/e_stop/reset` clears it. Latching is
deliberate -- an e-stop a retry can undo isn't an e-stop. This module never
commands motion itself; consumers are responsible for holding (stop
publishing targets, never a zero/home target) and aborting the active goal.
"""

import threading

from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Empty

# Reliable, depth-1, volatile durability -- deliberately not TRANSIENT_LOCAL.
# A retained Empty could never be retracted, so a node started after the
# e-stop press would latch forever. A late-starting node simply does not see
# a trip that happened before it started; the operator holds the physical
# situation until reset.
ESTOP_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
)


class EStopMonitor:
    """Subscribes to an e-stop topic and its reset topic; tracks a latched trip.

    - ``tripped``: True from the first e-stop message until a reset message
      (or :meth:`clear`) arrives.
    - ``on_estop``: optional callback invoked once per trip (not once per
      message), from the executor thread of the subscription. Must be cheap
      and non-blocking -- set a flag/event, never publish motion or block.
    - :meth:`trip` / :meth:`clear`: programmatic equivalents, also used by
      tests.
    """

    def __init__(self, node, topic='/e_stop', reset_topic='/e_stop/reset', on_estop=None):
        self._node = node
        self._log = node.get_logger()
        self._lock = threading.Lock()
        self._tripped = False
        self._on_estop = on_estop

        self._sub_estop = node.create_subscription(
            Empty, topic, self._estop_callback, ESTOP_QOS,
        )
        self._sub_reset = node.create_subscription(
            Empty, reset_topic, self._reset_callback, ESTOP_QOS,
        )

    @property
    def tripped(self):
        with self._lock:
            return self._tripped

    def _estop_callback(self, _msg):
        self.trip()

    def _reset_callback(self, _msg):
        self.clear()

    def trip(self):
        """Latch the trip. No-op (no re-logging, no re-firing callback) if already tripped."""
        with self._lock:
            already_tripped = self._tripped
            self._tripped = True
        if already_tripped:
            return
        self._log.warn('e-stop tripped; aborting autonomous actions and holding')
        if self._on_estop is not None:
            self._on_estop()

    def clear(self):
        """Clear the latch. No-op if not currently tripped."""
        with self._lock:
            was_tripped = self._tripped
            self._tripped = False
        if was_tripped:
            self._log.warn('e-stop reset; autonomous actions may resume')
