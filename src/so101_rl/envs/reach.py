"""So101Reach-v0: move the gripper tip to a goal point with the five arm joints.

The observation and the action are defined by `ReachIO` from joint positions only, so the same policy runs on
`MujocoFollower` here and on the real follower (through `SafeFollower`) later. The contract is in docs/reach.md;
changing it breaks policies trained before.

    import gymnasium as gym
    env = gym.make("so101_rl.envs:So101Reach-v0")

The env applies the same workspace rule as so101-core's `SafeFollower` (`so101.safety.leaves_workspace`): a step that
would take the arm out of the workspace from so101_config/safety.json (the table), narrowed by the env's own
`workspace`, is held. So `SafeFollower` never has to clip or hold this policy on the real arm.
"""

import time
from collections.abc import Mapping, Sequence
from typing import ClassVar

import gymnasium as gym
import numpy as np
from so101.conventions import GRIPPER
from so101.kinematics import ArmPoints
from so101.safety import SafetyLimits, Workspace, leaves_workspace
from so101.sim.follower import MujocoFollower

from so101_rl.sim import ARM_JOINTS, SUFFIX, Kinematics, install_scene

SCENE = "scene_reach.xml"
GRIPPER_TARGET = 0.0  # rad, about 10 % open; the gripper does not move the tip

# Observation layout (19 x float32): radians and metres in the base frame.
OBS_SIZE = 19
Q = slice(0, 5)  # measured arm joints
TARGET = slice(5, 10)  # last joint targets sent
EE = slice(10, 13)  # gripper tip, FK of the measured joints
GOAL = slice(13, 16)
GOAL_DELTA = slice(16, 19)  # goal - ee
# Bounds of the observation space: every joint range is within ±pi, and the arm reaches less than 0.5 m.
OBS_HIGH = np.concatenate([np.full(10, np.pi), np.full(6, 1.0), np.full(3, 2.0)]).astype(np.float32)

# Where goals may be, for the tip: in front of the base (x), above the table (z), away from the base column and within
# reach (r, the horizontal distance from the base's z axis). The whole arm must also be inside the workspace.
REACH_GOAL_REGION = Workspace(x_min=0.05, z_min=0.05, r_min=0.12, r_max=0.45)
GOAL_POSE_FRACTION = 0.7  # goals are the FK of poses within this fraction of the soft limits
RESET_RAMP_STEPS = 10  # send_action calls from home to the start pose
RESET_HOLD_STEPS = 5  # then holding it, so the arm starts nearly at rest
MAX_SAMPLE_TRIES = 1000


def arm_positions(obs: Mapping[str, float]) -> np.ndarray:
    """The five arm joint positions (rad) from an observation keyed `"<joint>.pos"`."""
    return np.array([obs[f"{joint}{SUFFIX}"] for joint in ARM_JOINTS], dtype=np.float64)


def _soft_limits(limits: SafetyLimits) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([limits.joint_limits[joint][0] for joint in ARM_JOINTS])
    upper = np.array([limits.joint_limits[joint][1] for joint in ARM_JOINTS])
    return lower, upper


def _pose(arm: np.ndarray, gripper: float = GRIPPER_TARGET) -> dict[str, float]:
    """Arm positions (5) plus the gripper as {joint: rad}."""
    return {joint: float(value) for joint, value in zip(ARM_JOINTS, arm, strict=True)} | {GRIPPER: gripper}


class ReachIO:
    """The reach task's observation and action, from joint positions only: the same in the sim and on the real arm.

    limits: the soft joint limits and the workspace (`SafetyLimits.load()` reads so101_config/safety.json).
    action_scale: rad per step for an action of 1. It must not exceed `max_step`, so `SafeFollower` never clips.
    workspace: the task's own workspace; it narrows `limits.workspace`, never widens it. `self.limits` has the result.
    kinematics: the tip FK; defaults to `Kinematics()` on the bare arm. The env passes one on its scene model.
    gripper: the fixed gripper target (rad).

        io.reset(obs)                      # targets := the measured joints
        robot.send_action(io.act(action))  # action in [-1, 1]^5 -> {"<joint>.pos": rad}, gripper included
        io.observe(robot.get_observation(), goal)  # 19 x float32
    """

    def __init__(
        self,
        limits: SafetyLimits,
        action_scale: float = 0.05,
        workspace: Workspace | None = None,
        kinematics: Kinematics | None = None,
        gripper: float = GRIPPER_TARGET,
    ):
        self.limits = limits.with_workspace(workspace)
        max_step = min(self.limits.max_step[joint] for joint in ARM_JOINTS)
        if not 0.0 < action_scale <= max_step:
            raise ValueError(
                f"action_scale must be in (0, {max_step}] (the smallest arm max_step in safety.json), so SafeFollower "
                f"never clips this policy; got {action_scale}."
            )
        gripper_lower, gripper_upper = self.limits.joint_limits[GRIPPER]
        if not gripper_lower <= gripper <= gripper_upper:
            raise ValueError(f"gripper target {gripper} is outside its soft limits [{gripper_lower}, {gripper_upper}].")
        self.action_scale = float(action_scale)
        self.gripper = float(gripper)
        self.lower, self.upper = _soft_limits(self.limits)
        self.kinematics = kinematics if kinematics is not None else Kinematics()
        self.arm = ArmPoints()
        self.targets: np.ndarray | None = None  # the five arm joint targets (rad)
        self.held = False  # the last act() was held at the workspace boundary
        self.ee: np.ndarray | None = None  # the tip position from the last observe()

    @property
    def workspace(self) -> Workspace:
        return self.limits.workspace

    def reset(self, obs: Mapping[str, float]) -> None:
        """Start from the measured joints: they become the targets (clamped into the soft limits)."""
        self.targets = np.clip(arm_positions(obs), self.lower, self.upper)
        self.held = False

    def observe(self, obs: Mapping[str, float], goal: Sequence[float]) -> np.ndarray:
        """`[q, target, ee, goal, goal - ee]` as a new 19 x float32 array."""
        self._check_reset()
        goal = np.asarray(goal, dtype=np.float64)
        self.ee = self.kinematics.ee_position(obs)
        return np.concatenate([arm_positions(obs), self.targets, self.ee, goal, goal - self.ee]).astype(np.float32)

    def act(self, action: Sequence[float]) -> dict[str, float]:
        """Action in [-1, 1]^5 (clipped) -> new targets as {"<joint>.pos": rad} for every joint, gripper included.

        Each target changes by `action * action_scale` and is clamped to the soft limits. When that would take the arm
        out of the workspace, all targets stay where they are and `held` is True (`so101.safety.leaves_workspace`).
        """
        self._check_reset()
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(len(ARM_JOINTS)), -1.0, 1.0)
        targets = np.clip(self.targets + action * self.action_scale, self.lower, self.upper)
        self.held = leaves_workspace(self.pose(), _pose(targets, self.gripper), self.workspace, self.arm)
        if not self.held:
            self.targets = targets
        return {f"{joint}{SUFFIX}": value for joint, value in self.pose().items()}

    def pose(self) -> dict[str, float]:
        """The current targets as {joint: rad}, gripper included."""
        self._check_reset()
        return _pose(self.targets, self.gripper)

    def _check_reset(self) -> None:
        if self.targets is None:
            raise RuntimeError("ReachIO has no targets yet; call reset(observation) first.")


def sample_goal_pose(
    rng: np.random.Generator,
    kinematics: Kinematics,
    limits: SafetyLimits,
    goal_region: Workspace = REACH_GOAL_REGION,
    arm: ArmPoints | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A random reachable goal: (arm pose (5, rad), goal = the tip position of that pose (3, m)).

    The pose is uniform within `GOAL_POSE_FRACTION` of the soft limits. It is accepted when the tip is inside
    `goal_region` and the whole arm (`ArmPoints`) inside `limits.workspace`, so the arm can hold it.
    """
    arm = arm if arm is not None else ArmPoints()
    lower, upper = _soft_limits(limits)
    for _ in range(MAX_SAMPLE_TRIES):
        pose = rng.uniform(GOAL_POSE_FRACTION * lower, GOAL_POSE_FRACTION * upper)
        q = _pose(pose)
        goal = kinematics.ee_position(q)
        if goal_region.contains(goal) and limits.workspace.arm_excess(arm.positions(q)) == 0.0:
            return pose, goal
    raise RuntimeError(f"No goal found in {MAX_SAMPLE_TRIES} tries: do goal_region and the workspace overlap?")


def sample_goal(
    rng: np.random.Generator,
    kinematics: Kinematics,
    limits: SafetyLimits,
    goal_region: Workspace = REACH_GOAL_REGION,
    arm: ArmPoints | None = None,
) -> np.ndarray:
    """A random reachable goal position (3, m), see `sample_goal_pose`."""
    return sample_goal_pose(rng, kinematics, limits, goal_region, arm)[1]


def sample_start(
    rng: np.random.Generator, limits: SafetyLimits, init_noise: float, arm: ArmPoints | None = None
) -> np.ndarray:
    """A random start pose (5, rad): home (every joint 0) ± `init_noise`, inside the soft limits and the workspace."""
    arm = arm if arm is not None else ArmPoints()
    lower, upper = _soft_limits(limits)
    for _ in range(MAX_SAMPLE_TRIES):
        pose = np.clip(rng.uniform(-init_noise, init_noise, len(ARM_JOINTS)), lower, upper)
        if limits.workspace.arm_excess(arm.positions(_pose(pose))) == 0.0:
            return pose
    raise RuntimeError(f"No start pose inside the workspace in {MAX_SAMPLE_TRIES} tries.")


class ReachEnv(gym.Env):
    """So101Reach-v0 on `MujocoFollower(scene="scene_reach.xml")`. See docs/reach.md for the contract.

    control_hz, episode_s: control rate and episode length; episodes are truncated, never terminated, so the policy
        learns to reach and hold.
    action_scale: rad per step for an action of 1 (`ReachIO`).
    success_m: the tip is within this distance of the goal (`info["is_success"]`).
    success_bonus, action_cost, workspace_cost, contact_cost: reward terms, see `step`. 0 turns a term off.
    init_noise: the start pose is home ± this (rad) per arm joint.
    limits: `SafetyLimits`; None loads so101_config/safety.json.
    workspace: the task's own workspace, narrowing the one in `limits` (sampling, and the hold in `ReachIO`).
    goal_region: where random goals may be, for the tip (and `options["goal"]` must be).
    render_mode: "human" opens the MuJoCo viewer and runs in real time.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        control_hz: int = 30,
        episode_s: float = 4.0,
        action_scale: float = 0.05,
        success_m: float = 0.02,
        success_bonus: float = 0.1,
        action_cost: float = 0.01,
        init_noise: float = 0.3,
        limits: SafetyLimits | None = None,
        workspace: Workspace | None = None,
        goal_region: Workspace = REACH_GOAL_REGION,
        workspace_cost: float = 0.1,
        contact_cost: float = 0.1,
        render_mode: str | None = None,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None, got {render_mode!r}")
        if init_noise < 0:
            raise ValueError(f"init_noise must be >= 0, got {init_noise}")
        self.metadata = {**self.metadata, "render_fps": control_hz}
        self.render_mode = render_mode
        self.control_hz = control_hz
        self.max_steps = round(episode_s * control_hz)
        self.success_m = success_m
        self.success_bonus = success_bonus
        self.action_cost = action_cost
        self.workspace_cost = workspace_cost
        self.contact_cost = contact_cost
        self.init_noise = init_noise
        self.goal_region = goal_region

        limits = limits if limits is not None else SafetyLimits.load()
        self.robot = MujocoFollower(control_hz=control_hz, viewer=render_mode == "human", scene=install_scene(SCENE))
        self.robot.connect()
        try:
            model = self.robot.model
            self.io = ReachIO(limits, action_scale, workspace, Kinematics(model))
            self._mocap = int(model.body("goal").mocapid[0])
            self._floor = model.geom("floor").id
            # Geoms that move with the arm: their body is not welded to the world (the base and the floor are).
            self._moving_geoms = np.flatnonzero(model.body_weldid[model.geom_bodyid] != 0)
        except BaseException:
            self.robot.disconnect()
            raise
        self.limits = self.io.limits  # with the task's workspace

        self.observation_space = gym.spaces.Box(-OBS_HIGH, OBS_HIGH, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(len(ARM_JOINTS),), dtype=np.float32)
        self.goal = np.zeros(3)
        self._steps = 0
        self._next_frame = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Home, a new goal (or `options["goal"]`), then drive to a random start pose with `send_action`.

        The start pose is drawn first, so a seed gives the same start pose with or without `options["goal"]`.
        """
        super().reset(seed=seed)
        options = options or {}
        self.robot.reset()
        start = sample_start(self.np_random, self.limits, self.init_noise, self.io.arm)
        if options.get("goal") is not None:
            self.goal = self._checked_goal(options["goal"])
        else:
            self.goal = sample_goal(self.np_random, self.io.kinematics, self.limits, self.goal_region, self.io.arm)
        self.robot.data.mocap_pos[self._mocap] = self.goal

        home = arm_positions(self.robot.get_observation())
        for i in range(1, RESET_RAMP_STEPS + 1):
            self.robot.send_action(self._action(home + (start - home) * i / RESET_RAMP_STEPS))
        for _ in range(RESET_HOLD_STEPS):
            self.robot.send_action(self._action(start))

        obs = self.robot.get_observation()
        self.io.reset(obs)
        observation = self.io.observe(obs, self.goal)
        self._steps = 0
        self._next_frame = time.perf_counter()
        return observation, self._info(held=False, contact=False)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """reward = -distance + success_bonus·[distance < success_m] - action_cost·‖a‖²
        - workspace_cost·[step held at the workspace] - contact_cost·[the arm touches the floor]"""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.robot.send_action(self.io.act(action))
        observation = self.io.observe(self.robot.get_observation(), self.goal)
        self._steps += 1

        held, contact = self.io.held, self._touches_floor()
        info = self._info(held, contact)
        reward = (
            -info["distance"]
            + self.success_bonus * info["is_success"]
            - self.action_cost * float(np.sum(action**2))
            - self.workspace_cost * held
            - self.contact_cost * contact
        )
        if self.render_mode == "human":
            self._pace()
        return observation, float(reward), False, self._steps >= self.max_steps, info

    def render(self) -> None:
        """Nothing to do: with render_mode="human" the viewer is synced on every step."""

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()

    def _action(self, arm: np.ndarray) -> dict[str, float]:
        return {f"{joint}{SUFFIX}": value for joint, value in _pose(arm, self.io.gripper).items()}

    def _checked_goal(self, goal) -> np.ndarray:
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError(f'options["goal"] must be three finite numbers [x, y, z] in metres, got {goal!r}')
        if not self.goal_region.contains(goal):
            raise ValueError(f'options["goal"] {goal.tolist()} is outside the goal region {self.goal_region}')
        return goal

    def _info(self, held: bool, contact: bool) -> dict:
        distance = float(np.linalg.norm(self.goal - self.io.ee))
        return {"distance": distance, "is_success": distance < self.success_m, "held": held, "contact": contact}

    def _touches_floor(self) -> bool:
        pairs = self.robot.data.contact.geom  # (ncon, 2) geom ids of the active contacts
        if len(pairs) == 0:
            return False
        on_floor = pairs == self._floor
        other = np.where(on_floor[:, 0], pairs[:, 1], np.where(on_floor[:, 1], pairs[:, 0], -1))
        return bool(np.isin(other, self._moving_geoms).any())

    def _pace(self) -> None:
        """Real time for the viewer: wait until the next frame is due."""
        self._next_frame += 1.0 / self.control_hz
        delay = self._next_frame - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_frame = time.perf_counter()
