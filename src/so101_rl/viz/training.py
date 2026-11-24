"""Rerun for `so101-rl-train`: what it passes to `so101_rl.training.train`.

    hooks = RerunTraining(cfg, limits, view=True)
    try:
        train(cfg, runs_dir, limits=limits, **hooks.train_kwargs())
    finally:
        hooks.close()

The recording goes to runs/<name>/rerun.rrd (and to the viewer with `view`): open it later with
`uv run rerun runs/<name>/rerun.rrd`.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
from so101.safety import SafetyLimits
from stable_baselines3.common.callbacks import BaseCallback

from so101_rl.training import RunConfig, make_env
from so101_rl.viz.blueprint import blueprint
from so101_rl.viz.metrics import RerunOutputFormat, TrainEpisodeCallback
from so101_rl.viz.recording import close, connect, new_recording, run_color, spawn_viewer
from so101_rl.viz.rollout import ROLLOUT_SEEDS, RolloutCallback

RRD_NAME = "rerun.rrd"

HOW_TO_READ = """\
## Reading this recording

- **Learning** and **Losses** are on the `train_step` timeline (env steps). *Eval success rate* is the share of
  deterministic eval episodes that end with the tip within the success distance.
- **Rollouts**: switch the time panel at the bottom to `rollout_frame` and press play. After every evaluation the
  policy is recorded on the same fixed seeds, so you watch the same goals and start poses at every checkpoint. The red
  sphere is the success zone around the goal; the tip turns green inside it. The episode log names the checkpoint.
- *Fixed-seed rollouts: final distance* (Learning) shows those same episodes over training, one line per seed.
"""


def _value(value) -> str:
    if value in ({}, None):
        return "-"
    return f"`{json.dumps(value)}`" if isinstance(value, (dict, list)) else str(value)


def run_markdown(config: dict) -> str:
    """A run's config.json (and result, when there is one) as a markdown page."""
    result = config.get("result") or {}
    rows = [
        ("Env", f"`{config['env_id']}`"),
        ("Env kwargs", _value(config["env_kwargs"])),
        ("Algorithm", config["algo"].upper()),
        ("Algorithm kwargs", _value(config["algo_kwargs"])),
        ("Seed", config["seed"]),
        ("Parallel envs", config["n_envs"]),
        ("Steps", f"{config['steps']:,}"),
        ("Evaluation", f"every {config['eval']['every']:,} steps, {config['eval']['episodes']} episodes"),
        ("Device", config["device"]),
        ("Started", config["started"]),
        ("so101-rl", f"`{config['git'].get('so101-rl')}`"),
        ("so101-core", f"`{config['git'].get('so101-core')}`"),
    ]
    if result:
        rows += [
            ("Steps done", f"{result['steps_done']:,}" + (" (interrupted)" if result["interrupted"] else "")),
            ("Wall time", f"{result['wall_time_s'] / 60:.1f} min"),
            ("Eval success, last / best", _rates(result)),
        ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    limits = json.dumps(config["safety_limits"], indent=2)
    return (
        f"# {config['name']}\n\n| | |\n|---|---|\n{table}\n\n{HOW_TO_READ}\n"
        f"## Safety limits used\n\n```json\n{limits}\n```\n"
    )


def _rates(result: dict) -> str:
    last, best = result.get("last_success_rate"), result.get("best_success_rate")
    if last is None:
        return "-"
    return f"{last:.0%} / {best:.0%}"


def log_run_page(rec: rr.RecordingStream, run: Path) -> None:
    config = run / "config.json"
    if config.is_file():
        text = run_markdown(json.loads(config.read_text()))
        rec.log("run", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True)


class StartRecording(BaseCallback):
    """Connects the recording to runs/<name>/rerun.rrd (and the viewer) when training starts.

    `train` makes the run folder, so the file can only be opened then: the SB3 logger's folder is the run folder.
    """

    def __init__(self, rec: rr.RecordingStream, view: bool, layout: rrb.Blueprint):
        super().__init__()
        self.rec = rec
        self.view = view
        self.layout = layout
        self.run: Path | None = None

    def _init_callback(self) -> None:
        self.run = Path(self.logger.get_dir())
        connect(self.rec, self.run / RRD_NAME, self.view, self.layout)
        log_run_page(self.rec, self.run)

    def _on_step(self) -> bool:
        return True


class RerunTraining:
    """The Rerun side of one training run: an SB3 logger output, callbacks, and the rollouts after each evaluation.

    view: spawn the viewer now and stream to it live (the file is written either way).
    seeds: the fixed rollout seeds (same goals and start poses at every checkpoint).
    """

    def __init__(self, cfg: RunConfig, limits: SafetyLimits, view: bool = False, seeds: Sequence[int] = ROLLOUT_SEEDS):
        self.rec = new_recording(cfg.name)
        if view:
            spawn_viewer(self.rec)
        color = run_color(0)
        self.output_format = RerunOutputFormat(self.rec, cfg.name, color)
        self.start = StartRecording(self.rec, view, blueprint([cfg.algo]))
        self.episodes = TrainEpisodeCallback(self.rec, cfg.name, color)
        self.rollouts = RolloutCallback(
            self.rec, cfg.name, lambda: make_env(cfg.env, cfg.env_kwargs, limits), seeds, color
        )

    def train_kwargs(self) -> dict:
        return {
            "output_formats": [self.output_format],
            "callbacks": [self.start, self.episodes],
            "after_eval": self.rollouts,
        }

    def close(self) -> None:
        """Close the rollout env; update the Run page with the result; finish the file."""
        self.rollouts.close()
        if self.start.run is not None:
            log_run_page(self.rec, self.start.run)
        close(self.rec)
