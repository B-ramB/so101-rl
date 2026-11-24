"""Rerun recordings for training runs: one per run, its sinks, colors and entity names."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

APP_ID = "so101_rl"

TRAIN_STEP = "train_step"
ROLLOUT_FRAME = "rollout_frame"
EPISODE_TIME = "episode_time"

# The validated categorical palette (docs: dataviz reference palette), dark steps because the Rerun viewer is dark.
# Fixed order, never cycled: run i keeps slot i in every view, so at most len(RUN_COLORS) runs are compared at once.
RUN_COLORS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
GOOD = "#0ca30c"  # status: success
CRITICAL = "#d03b3b"  # status: failure
FLUSH_TIMEOUT_S = 10.0
GOAL = (230, 51, 51, 90)  # the goal marker of scene_reach.xml (rgba 0.9 0.2 0.2), see-through


def rgb(color: str) -> tuple[int, int, int]:
    """Hex to RGB: "#3987e5" -> (57, 135, 229)."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def run_color(index: int) -> tuple[int, int, int]:
    """The color of the run in slot `index`."""
    if not 0 <= index < len(RUN_COLORS):
        raise ValueError(f"at most {len(RUN_COLORS)} runs can be shown at once (one color each), got slot {index}")
    return rgb(RUN_COLORS[index])


def entity_name(name: str) -> str:
    """A name as a single entity path part: anything but letters, digits, `-` and `_` becomes `_`."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name) or "_"


def new_recording(name: str) -> rr.RecordingStream:
    """A recording for the run `name`, without sinks yet: what is logged is buffered until `connect`."""
    rec = rr.RecordingStream(APP_ID, recording_id=name)
    rec.send_recording_name(name)
    rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # the arm's base frame: x forward, z up
    return rec


def spawn_viewer(rec: rr.RecordingStream) -> None:
    """Start the Rerun viewer for `rec` (or reuse a running one) without connecting yet: `connect` does that.

    Through the recording: the module-level `rr.spawn()` checks the active recording, and without `rr.init` there is
    none, so it would skip the spawn ("Rerun is disabled - spawn() call ignored").
    """
    rec.spawn(connect=False)


def connect(rec: rr.RecordingStream, rrd: Path | None, view: bool, blueprint: rrb.Blueprint) -> None:
    """Send `rec` to the file `rrd` and/or a viewer (`spawn_viewer` first), with `blueprint` as its layout."""
    sinks: list = []
    if view:
        sinks.append(rr.GrpcSink())
    if rrd is not None:
        sinks.append(rr.FileSink(str(rrd)))
    if not sinks:
        raise ValueError("nowhere to send the recording: pass a file, view=True, or both")
    rec.set_sinks(*sinks, default_blueprint=blueprint)


def close(rec: rr.RecordingStream) -> None:
    """Flush everything to the sinks and finish the file.

    When the viewer cannot be reached (it was closed, or never started), the flush fails after a few seconds; the file
    sink is complete anyway, so that is only reported.
    """
    try:
        rec.flush(timeout_sec=FLUSH_TIMEOUT_S)
    except RuntimeError as e:
        print(f"Rerun viewer not reached ({e}); the .rrd file is complete.", flush=True)
    rec.disconnect()


@contextmanager
def active(rec: rr.RecordingStream) -> Iterator[None]:
    """Make `rec` this thread's recording, for code that logs through `rr.log` (so101-core's `ArmModel`).

    Unlike `with rec:`, this does not finalize the file sink on exit. Use it for one-off setup only: every swap of the
    thread-local recording flushes Rerun's batcher, so per-frame use writes one chunk per entity per frame.
    """
    previous = rr.set_thread_local_data_recording(rec)
    try:
        yield
    finally:
        rr.set_thread_local_data_recording(previous)
