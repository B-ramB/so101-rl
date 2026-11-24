"""Episodes of a policy in Rerun: the arm moving, the goal, the tip trail, and per-step plots.

`play` runs one seeded episode for one or more `Player`s in lockstep, so several policies can be watched side by side
on the same goal. `RolloutCallback` uses it during training: after each evaluation it records the same fixed seeds,
so every checkpoint (and every run) is shown on the same goals and start poses.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import rerun as rr
from so101.conventions import GRIPPER, GRIPPER_LOWER, GRIPPER_UPPER, rescale_gripper
from so101.paths import FOLLOWER_URDF
from so101.viz.rerun_arm import ArmModel
from stable_baselines3.common.callbacks import BaseCallback

from so101_rl.envs.reach import EE, Q
from so101_rl.sim import ARM_JOINTS
from so101_rl.viz.recording import (
    CRITICAL,
    EPISODE_TIME,
    GOAL,
    GOOD,
    ROLLOUT_FRAME,
    RUN_COLORS,
    TRAIN_STEP,
    active,
    entity_name,
    rgb,
)

ROLLOUT_SEEDS = (1000, 1001, 1002)
TIP_RADIUS = 0.006  # m
TRAIL_RADIUS = 0.002  # m
LABEL_HEIGHT = 0.5  # m above the base, where an arm's name is shown


def rollout_seeds(n: int) -> tuple[int, ...]:
    """The first `n` fixed rollout seeds: 1000, 1001, ..."""
    return tuple(range(ROLLOUT_SEEDS[0], ROLLOUT_SEEDS[0] + n))


class RolloutRecorder:
    """Logs the episodes of one arm in a reach env.

    arm: entity name of the arm (one path part). The URDF, goal, tip and trail go under it.
    series: this arm's name in the plots (usually the run).
    plots: the entity root of the plots: `<plots>/distance/<series>`, `<plots>/reward/<series>` and `<plots>/log`.
    offset: where the arm stands (m, world frame); everything 3D is shifted by it, so arms can stand side by side.
    color: the arm's trail and lines. detail: also plot the reward terms and actions (`<plots>/reward_terms/<term>`,
        `<plots>/action/<joint>`); for one arm at a time.
    label: shown above the arm.
    """

    def __init__(
        self,
        rec: rr.RecordingStream,
        arm: str = "rollout",
        series: str = "policy",
        plots: str = "rollout_plots",
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        color: tuple[int, int, int] = rgb(RUN_COLORS[0]),
        detail: bool = True,
        label: str | None = None,
    ):
        if entity_name(arm) != arm:
            raise ValueError(f"arm must be a single entity path part, got {arm!r}")
        self.rec = rec
        self.arm = arm
        self.series = entity_name(series)
        self.plots = plots
        self.offset = np.asarray(offset, dtype=np.float64)
        self.color = color
        self.detail = detail
        with active(rec):
            self.model = ArmModel(arm, FOLLOWER_URDF)
            self.model.log_placement(self.offset.tolist())
        if label is not None:
            position = self.offset + (0.0, 0.0, LABEL_HEIGHT)
            rec.log(f"{arm}/label", rr.Points3D([position], labels=[label], colors=[color], radii=[0.0]), static=True)
        style = rr.SeriesLines(names=series, colors=[color], widths=[2.0])
        rec.log(f"{plots}/distance/{self.series}", style, static=True)
        rec.log(f"{plots}/reward/{self.series}", style, static=True)
        self._trail: list[np.ndarray] = []
        self._gripper = 0.0

    def begin(self, env: gym.Env, title: str) -> None:
        """A new episode, after `env.reset()`: the goal moves, the trail restarts."""
        reach = env.unwrapped
        self._trail = []
        self._gripper = float(reach.io.gripper)
        goal = np.asarray(reach.goal) + self.offset
        self.rec.log(f"{self.arm}/goal", rr.Points3D([goal], radii=[reach.success_m], colors=[GOAL]))
        self.rec.log(f"{self.plots}/log", rr.TextLog(f"{self.series}: {title}", level="INFO"))

    def frame(self, obs: np.ndarray, info: dict, action: np.ndarray | None = None, reward: float | None = None) -> None:
        """One step: the arm from the measured joints in `obs`, the tip from its FK (as the policy sees them)."""
        joints = {joint: float(value) for joint, value in zip(ARM_JOINTS, obs[Q], strict=True)}
        self._log_joints(joints | {GRIPPER: self._gripper})
        tip = np.asarray(obs[EE], dtype=np.float64) + self.offset
        self._trail.append(tip)
        tip_color = rgb(GOOD) if info.get("is_success") else self.color
        self.rec.log(f"{self.arm}/tip", rr.Points3D([tip], radii=[TIP_RADIUS], colors=[tip_color]))
        self.rec.log(f"{self.arm}/trail", rr.LineStrips3D([self._trail], radii=[TRAIL_RADIUS], colors=[self.color]))
        self.rec.log(f"{self.plots}/distance/{self.series}", rr.Scalars(float(info["distance"])))
        if reward is not None:
            self.rec.log(f"{self.plots}/reward/{self.series}", rr.Scalars(float(reward)))
        if self.detail:
            for term, value in info.get("reward_terms", {}).items():
                self.rec.log(f"{self.plots}/reward_terms/{term}", rr.Scalars(float(value)))
            if action is not None:
                for joint, value in zip(ARM_JOINTS, np.asarray(action).reshape(-1), strict=True):
                    self.rec.log(f"{self.plots}/action/{joint}", rr.Scalars(float(value)))

    def _log_joints(self, positions: dict[str, float]) -> None:
        """`ArmModel.log_positions`, but through `self.rec` instead of the active recording.

        Making a recording active swaps the thread-local recording, and every swap flushes Rerun's batcher: doing that
        each frame wrote one chunk per entity per frame, about twice the file size.
        """
        for joint, value in positions.items():
            if joint == GRIPPER:
                value = rescale_gripper(value, (GRIPPER_LOWER, GRIPPER_UPPER), self.model.gripper_limits)
            if joint in self.model.joints:
                transform = self.model.joints[joint].compute_transform(value)
                self.rec.log(f"{self.arm}/joint_transforms/{joint}", transform)

    def end(self, info: dict) -> str:
        """The episode's last step: logs and returns a one-line summary."""
        distance, success = float(info["distance"]), bool(info.get("is_success"))
        summary = f"{self.series}: final distance {distance * 100:.1f} cm, {'reached' if success else 'missed'}"
        self.rec.log(
            f"{self.plots}/log", rr.TextLog(summary, level="INFO", color=rgb(GOOD) if success else rgb(CRITICAL))
        )
        return summary


@dataclass
class Player:
    """A policy in its own env, recorded by its own recorder."""

    recorder: RolloutRecorder
    env: gym.Env
    policy: Callable[[np.ndarray], np.ndarray]  # observation -> action


def sb3_policy(model) -> Callable[[np.ndarray], np.ndarray]:
    """An SB3 model as a deterministic policy."""
    return lambda obs: model.predict(obs, deterministic=True)[0]


def _set_time(rec: rr.RecordingStream, frame: int, seconds: float, train_step: int | None) -> None:
    rec.reset_time()
    rec.set_time(ROLLOUT_FRAME, sequence=frame)
    rec.set_time(EPISODE_TIME, duration=seconds)
    if train_step is not None:
        rec.set_time(TRAIN_STEP, sequence=train_step)


def play(
    rec: rr.RecordingStream,
    players: Sequence[Player],
    seed: int,
    frame: int,
    train_step: int | None = None,
    title: str = "",
) -> tuple[int, list[dict]]:
    """One episode with `seed` for every player, in lockstep, starting at `rollout_frame` = `frame`.

    The same seed gives the same goal and start pose in every player's env (when their env kwargs agree). Frames also
    carry `train_step` when given (the checkpoint being shown). Returns the next free frame and each player's last info.
    """
    hz = players[0].env.unwrapped.control_hz
    observations, infos = [], []
    _set_time(rec, frame, 0.0, train_step)
    for player in players:
        obs, info = player.env.reset(seed=seed)
        player.recorder.begin(player.env, f"seed {seed}" + (f", {title}" if title else ""))
        player.recorder.frame(obs, info)
        observations.append(obs)
        infos.append(info)
    done = [False] * len(players)
    step = 0
    while not all(done):
        step += 1
        _set_time(rec, frame + step, step / hz, train_step)
        for i, player in enumerate(players):
            if done[i]:
                continue
            action = player.policy(observations[i])
            obs, reward, terminated, truncated, info = player.env.step(action)
            player.recorder.frame(obs, info, action, reward)
            observations[i], infos[i] = obs, info
            done[i] = terminated or truncated
    for player, info in zip(players, infos, strict=True):
        player.recorder.end(info)
    return frame + step + 1, infos


class RolloutCallback(BaseCallback):
    """Records the current policy on the fixed `seeds` after each evaluation (EvalCallback's `callback_after_eval`).

    Also logs each seed's final distance at `rollouts/final_distance/seed_<n>` on `train_step`, so the Learning view
    shows how the policy does on the very same goals over training. The env is made lazily by `make_env`, in this
    process; `close()` closes it.
    """

    def __init__(
        self,
        rec: rr.RecordingStream,
        run: str,
        make_env: Callable[[], gym.Env],
        seeds: Sequence[int] = ROLLOUT_SEEDS,
        color: tuple[int, int, int] = rgb(RUN_COLORS[0]),
    ):
        super().__init__()
        if len(seeds) > len(RUN_COLORS):
            raise ValueError(f"at most {len(RUN_COLORS)} rollout seeds (one color each), got {len(seeds)}")
        self.rec = rec
        self.run = run
        self.make_env = make_env
        self.seeds = tuple(seeds)
        self.color = color
        self.env: gym.Env | None = None
        self.player: Player | None = None
        self.frame = 0

    def _on_step(self) -> bool:
        if self.player is None:
            self.env = self.make_env()
            recorder = RolloutRecorder(self.rec, series=self.run, color=self.color)
            self.player = Player(recorder, self.env, sb3_policy(self.model))
            for i, seed in enumerate(self.seeds):
                style = rr.SeriesLines(names=f"seed {seed}", colors=[rgb(RUN_COLORS[i])], widths=[2.0])
                self.rec.log(f"rollouts/final_distance/seed_{seed}", style, static=True)
        step = self.num_timesteps
        for seed in self.seeds:
            self.frame, (info,) = play(self.rec, [self.player], seed, self.frame, step, f"step {step:,}")
            self.rec.reset_time()
            self.rec.set_time(TRAIN_STEP, sequence=step)
            self.rec.log(f"rollouts/final_distance/seed_{seed}", rr.Scalars(float(info["distance"])))
        return True

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
