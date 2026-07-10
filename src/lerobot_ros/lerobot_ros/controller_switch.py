"""Scoped ros2_control controller switching for autonomy operations.

An operation (e.g. an ArUco alignment or a policy run) activates the controller
it needs for its duration and restores the previously-active controller(s) when
it finishes -- on every exit path. This removes a class of orchestration errors
("the workflow forgot to switch / forgot to switch back") without changing the
safety model: motion is still gated entirely by the operator heartbeat and by
which node is allowed to publish, and any active controller with no input simply
holds. Switching only decides *which* gated input actuates.

Disabled (a pure no-op) unless ``activate_controller`` is set, so the default
behaviour is unchanged and robot-agnostic -- ``controller_manager_msgs`` is only
imported when switching is actually enabled.

NOTE: the controller_manager is assumed un-namespaced (``/controller_manager``).
If it is ever namespaced, pass the namespaced name via the ``controller_manager``
parameter -- see the matching note in the manipulator bringup repo.
"""

import time


class ControllerSwitcher:
    def __init__(self, node, activate_controller, exclusive_controllers,
                 controller_manager='/controller_manager', timeout_s=5.0):
        self._node = node
        self._log = node.get_logger()
        self.activate_controller = activate_controller or ''
        self.exclusive_controllers = [c for c in (exclusive_controllers or []) if c]
        self._cm = (controller_manager or '/controller_manager').rstrip('/')
        self._timeout = float(timeout_s)
        self._switch_cli = None
        self._list_cli = None
        self._SwitchController = None
        self._ListControllers = None
        if self.enabled:
            self._make_clients()

    @property
    def enabled(self):
        return bool(self.activate_controller)

    def _relevant(self):
        seen, out = set(), []
        for c in [self.activate_controller, *self.exclusive_controllers]:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _make_clients(self):
        try:
            from controller_manager_msgs.srv import ListControllers, SwitchController
        except ImportError as e:  # pragma: no cover - only on a ros2_control-less host
            self._log.error(
                f"controller switching enabled (activate_controller="
                f"'{self.activate_controller}') but controller_manager_msgs is "
                f"unavailable: {e}. Goals that need the switch will be aborted; "
                "install ros2_control or clear activate_controller.")
            return
        self._SwitchController = SwitchController
        self._ListControllers = ListControllers
        self._switch_cli = self._node.create_client(
            SwitchController, f"{self._cm}/switch_controller")
        self._list_cli = self._node.create_client(
            ListControllers, f"{self._cm}/list_controllers")

    def _call(self, client, request):
        if client is None:
            return None
        if not client.wait_for_service(timeout_sec=self._timeout):
            self._log.error(f"controller_manager service '{client.srv_name}' unavailable")
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + self._timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            self._log.error(f"controller_manager service '{client.srv_name}' timed out")
            return None
        return future.result()

    def _active_controllers(self):
        resp = self._call(self._list_cli, self._ListControllers.Request())
        if resp is None:
            return None
        return {c.name for c in resp.controller if c.state == 'active'}

    def _switch(self, activate, deactivate):
        req = self._SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        # BEST_EFFORT so activating an already-active controller (or deactivating
        # an already-inactive one) is a no-op rather than an error.
        req.strictness = self._SwitchController.Request.BEST_EFFORT
        resp = self._call(self._switch_cli, req)
        return bool(resp and resp.ok)

    def acquire(self):
        """Activate the operation controller for the duration of the operation.

        Returns ``(True, token)`` on success -- ``token`` is a snapshot of the
        relevant controllers that were active before, opaque to the caller and
        passed back to :meth:`release`. Returns ``(True, None)`` when disabled
        (no-op), and ``(False, None)`` on failure (caller should abort the goal).
        """
        if not self.enabled:
            return True, None
        if self._switch_cli is None or self._list_cli is None:
            return False, None
        active = self._active_controllers()
        if active is None:
            return False, None
        relevant = self._relevant()
        snapshot = [c for c in relevant if c in active]  # active-before, to restore
        activate = [] if self.activate_controller in active else [self.activate_controller]
        deactivate = [c for c in relevant if c in active and c != self.activate_controller]
        if activate or deactivate:
            if not self._switch(activate, deactivate):
                self._log.error(f"failed to activate controller '{self.activate_controller}'")
                return False, None
            self._log.info(
                f"activated '{self.activate_controller}' "
                f"(deactivated {deactivate or 'nothing'})")
        return True, snapshot

    def release(self, token):
        """Restore the controllers to their pre-:meth:`acquire` state.

        Best-effort and safe on all exit paths. ``token`` is what ``acquire``
        returned; ``None`` (disabled) is a no-op.
        """
        if not self.enabled or token is None or self._switch_cli is None:
            return
        relevant = self._relevant()
        activate = list(token)
        deactivate = [c for c in relevant if c not in token]
        if self._switch(activate, deactivate):
            self._log.info(
                f"restored controllers {activate or 'none'} "
                f"(deactivated {deactivate or 'nothing'})")
        else:
            self._log.warn("failed to restore controllers on operation exit")
