"""Project-local GLFW setup and deterministic passive-viewer shutdown."""

import os
import queue
import sys
import threading


def configure_backend() -> None:
    """Select the X11 wheel before importing MuJoCo/GLFW on Linux desktops."""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        if not os.environ.get("PYGLFW_LIBRARY"):
            os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")


def _prepare_viewer():
    import glfw
    import mujoco.viewer

    if (sys.platform.startswith("linux")
            and os.environ.get("PYGLFW_LIBRARY_VARIANT") == "x11"
            and glfw.platform_supported(glfw.PLATFORM_X11)):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    # Check on the caller thread: upstream launch_passive otherwise waits
    # forever for a handle if GLFW initialization fails in its render thread.
    if not glfw.init():
        _, description = glfw.get_error()
        detail = description.decode(errors="replace") if description else "unknown error"
        raise RuntimeError(
            f"GLFW initialization failed: {detail}. "
            "Run from a desktop terminal; control examples also support --headless."
        )
    return mujoco.viewer


def launch(model, data) -> None:
    """Blocking viewer used by the model-inspection example."""
    _prepare_viewer().launch(model, data)


class PassiveViewer:
    """Own the render thread so close() waits before GLFW can be terminated."""

    def __init__(self, model, data):
        viewer = _prepare_viewer()
        self._thread = None
        if sys.platform == "darwin":
            self._handle = viewer.launch_passive(model, data)
            return

        # MuJoCo's public Handle exposes no join operation. Use the same
        # launcher as launch_passive, but retain ownership of its thread.
        results = queue.Queue()

        def render() -> None:
            try:
                viewer._launch_internal(
                    model, data, run_physics_thread=False, handle_return=results
                )
            except Exception as exc:
                results.put(exc)

        self._thread = threading.Thread(target=render, name="xarm6-viewer", daemon=True)
        self._thread.start()
        result = results.get()
        if isinstance(result, Exception):
            self._thread.join()
            raise RuntimeError(f"MuJoCo viewer startup failed: {result}") from result
        self._handle = result

    def is_running(self) -> bool:
        return self._handle.is_running()

    def sync(self) -> None:
        self._handle.sync()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            if self._thread is not None:
                self._thread.join()
