"""Training curves in Rerun: SB3's logged metrics, and the final distance of every training episode."""

import math
from numbers import Real
from typing import Any

import rerun as rr
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from so101_rl.viz.recording import TRAIN_STEP, entity_name

FORMAT_NAME = "rerun"  # for SB3's `logger.record(..., exclude="rerun")`
EPISODE_MARKER_SIZE = 2.5  # thousands of training episodes: small markers stay readable


def metric_path(key: str, run: str) -> str:
    """`eval/success_rate` of run `r` -> `metrics/eval/success_rate/r`: one view per metric shows every run."""
    parts = [entity_name(part) for part in key.split("/") if part]
    return "/".join(["metrics", *parts, entity_name(run)])


def episode_path(run: str) -> str:
    return f"episodes/final_distance/{entity_name(run)}"


def is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _excluded(excluded: Any) -> bool:
    if excluded is None:
        return False
    if isinstance(excluded, str):
        return excluded == FORMAT_NAME
    return FORMAT_NAME in excluded


class RerunOutputFormat(KVWriter):
    """An SB3 logger output: each numeric metric goes to `metric_path(key, run)` on the `train_step` timeline.

    Strings and other non-numbers are skipped. Add it next to SB3's own outputs:
    `Logger(folder, [make_output_format("stdout", folder), RerunOutputFormat(rec, run, color)])`.
    """

    def __init__(self, rec: rr.RecordingStream, run: str, color: tuple[int, int, int]):
        self.rec = rec
        self.run = run
        self.color = color
        self._styled: set[str] = set()

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, Any], step: int = 0) -> None:
        self.rec.reset_time()  # rollout timelines must not leak onto the curves
        self.rec.set_time(TRAIN_STEP, sequence=int(step))
        for key, value in key_values.items():
            if _excluded(key_excluded.get(key)) or not is_number(value):
                continue
            path = metric_path(key, self.run)
            if path not in self._styled:
                self.rec.log(path, rr.SeriesLines(names=self.run, colors=[self.color], widths=[2.0]), static=True)
                self._styled.add(path)
            self.rec.log(path, rr.Scalars(float(value)))

    def close(self) -> None:
        """Nothing to do: the recording belongs to whoever created it."""


class TrainEpisodeCallback(BaseCallback):
    """Logs the final tip-to-goal distance (`info["distance"]`) of every finished training episode as a point.

    SB3's rollout/ep_rew_mean is a mean over the last 100 episodes; these points show the spread behind it.
    """

    def __init__(self, rec: rr.RecordingStream, run: str, color: tuple[int, int, int]):
        super().__init__()
        self.rec = rec
        self.path = episode_path(run)
        self.run = run
        self.color = color
        self._styled = False

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            if "episode" not in info or "distance" not in info:  # Monitor adds "episode" when one ends
                continue
            if not self._styled:
                points = rr.SeriesPoints(names=self.run, colors=[self.color], marker_sizes=[EPISODE_MARKER_SIZE])
                self.rec.log(self.path, points, static=True)
                self._styled = True
            self.rec.reset_time()
            self.rec.set_time(TRAIN_STEP, sequence=self.num_timesteps)
            self.rec.log(self.path, rr.Scalars(float(info["distance"])))
        return True
