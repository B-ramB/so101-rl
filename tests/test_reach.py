"""So101Reach-v0 and ReachIO: the Gymnasium API, the action contract (soft limits, step size, workspace), seeding.

Needs the URDFs (skipped otherwise). conftest.py points SO101_HOME at a temp dir, so safety.json there holds the
package defaults (workspace z_min = 0.01 m).
"""

import time
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from so101.conventions import JOINTS
from so101.kinematics import ArmPoints
from so101.paths import FOLLOWER_URDF
from so101.safety import SafeFollower, SafetyLimits, Workspace
from so101.sim.follower import MujocoFollower

from so101_rl.envs.reach import (
    EE,
    GOAL,
    GOAL_DELTA,
    OBS_SIZE,
    REACH_GOAL_REGION,
    SCENE,
    TARGET,
    Q,
    ReachEnv,
    ReachIO,
    sample_goal_pose,
    sample_start,
)
from so101_rl.sim import ARM_JOINTS, Kinematics, install_scene

pytestmark = pytest.mark.sim

TOL = 1e-6  # float32 observations


@pytest.fixture(scope="module")
def urdf():
    if not FOLLOWER_URDF.exists():
        pytest.skip(f"no URDF at {FOLLOWER_URDF}; run `uv run so101-fetch-urdfs`")
    return FOLLOWER_URDF


@pytest.fixture(scope="module")
def env(urdf):
    env = ReachEnv()
    yield env
    env.close()


def action_sequence(seed=0, n_random=150, n_saturated=150):
    """Random actions, then every joint at +1 and at -1: random walks never get near the limits or the table."""
    rng = np.random.default_rng(seed)
    yield from rng.uniform(-1, 1, (n_random, len(ARM_JOINTS)))
    yield from [np.ones(len(ARM_JOINTS))] * n_saturated
    yield from [-np.ones(len(ARM_JOINTS))] * n_saturated


def arm_inside(workspace, pose) -> bool:
    return workspace.arm_excess(ArmPoints().positions(pose)) == 0.0


@pytest.mark.filterwarnings("error")
def test_check_env(urdf):
    import so101_rl.envs  # noqa: F401  registers So101Reach-v0

    env = gym.make("So101Reach-v0")
    try:
        assert env.observation_space.shape == (OBS_SIZE,)
        assert env.action_space.shape == (len(ARM_JOINTS),)
        # The render check would open the MuJoCo viewer, which needs a display.
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_observation_layout(env):
    obs, info = env.reset(seed=3)
    assert obs.dtype == np.float32 and obs.shape == (OBS_SIZE,)
    measured = env.robot.get_observation()
    np.testing.assert_allclose(obs[Q], [measured[f"{joint}.pos"] for joint in ARM_JOINTS], atol=TOL)
    np.testing.assert_allclose(obs[TARGET], env.io.targets, atol=TOL)
    np.testing.assert_allclose(obs[EE], Kinematics(env.robot.model).ee_position(measured), atol=TOL)
    np.testing.assert_allclose(obs[GOAL], env.goal, atol=TOL)
    np.testing.assert_allclose(obs[GOAL_DELTA], obs[GOAL] - obs[EE], atol=TOL)
    assert info["distance"] == pytest.approx(np.linalg.norm(env.goal - obs[EE]), abs=TOL)


def test_targets_stay_inside_the_limits_and_the_workspace(env):
    obs, _ = env.reset(seed=0)
    lower, upper = env.io.lower, env.io.upper
    gripper = env.robot.model.actuator("gripper").id
    previous, held = obs[TARGET].astype(np.float64), 0
    for action in action_sequence():
        obs, *_, info = env.step(action)
        assert env.observation_space.contains(obs)
        targets = obs[TARGET].astype(np.float64)
        assert np.all(targets >= lower - TOL) and np.all(targets <= upper + TOL)
        assert np.all(np.abs(targets - previous) <= env.io.action_scale + TOL)
        assert arm_inside(env.limits.workspace, env.io.pose())
        assert env.robot.data.ctrl[gripper] == 0.0
        held += info["held"]
        previous = targets
    assert held > 0  # the saturated actions reached the workspace boundary


def test_safefollower_never_clips_the_reach_targets(urdf):
    """The contract: on any follower, SafeFollower passes ReachIO's commands unchanged (no clip, no hold)."""
    limits = SafetyLimits.load()
    with SafeFollower(MujocoFollower(scene=install_scene(SCENE)), limits) as robot:
        io = ReachIO(limits, kinematics=Kinematics(robot.robot.model))
        io.reset(robot.get_observation())
        held = 0
        for action in action_sequence(seed=1):
            robot.send_action(io.act(action))
            assert robot.last_clipped == {}
            assert not robot.last_held
            held += io.held
    assert held > 0


def test_a_move_into_the_table_is_held_and_penalised(urdf):
    env = ReachEnv(action_cost=0.0, contact_cost=0.0)
    try:
        env.reset(seed=0, options={"goal": [0.25, 0.0, 0.15]})
        bend = np.zeros(len(ARM_JOINTS))
        bend[ARM_JOINTS.index("elbow_flex")] = 1.0  # swings the gripper down into the table
        for _ in range(60):
            _, reward, *_, info = env.step(bend)
            if info["held"]:
                break
        assert info["held"]
        assert arm_inside(env.limits.workspace, env.io.pose())
        expected = -info["distance"] + env.success_bonus * info["is_success"] - env.workspace_cost
        assert reward == pytest.approx(expected)
    finally:
        env.close()


def test_floor_contact_is_detected_and_penalised(urdf):
    """Without a workspace nothing holds the arm, so it is driven into the floor; the contact costs contact_cost."""
    no_workspace = replace(SafetyLimits.load(), workspace=Workspace())
    env = ReachEnv(limits=no_workspace, action_cost=0.0, workspace_cost=0.0)
    try:
        env.reset(seed=0)
        bend = np.zeros(len(ARM_JOINTS))
        bend[ARM_JOINTS.index("elbow_flex")] = 1.0
        bend[ARM_JOINTS.index("shoulder_lift")] = 1.0
        for _ in range(60):
            _, reward, *_, info = env.step(bend)
            if info["contact"]:
                break
        assert info["contact"] and not info["held"]
        expected = -info["distance"] + env.success_bonus * info["is_success"] - env.contact_cost
        assert reward == pytest.approx(expected)
    finally:
        env.close()


def test_a_task_workspace_narrows_safety_json(urdf):
    env = ReachEnv(workspace=Workspace(z_min=0.08, r_max=0.4))
    try:
        assert env.limits.workspace == Workspace(z_min=0.08, r_max=0.4)  # stricter than safety.json's z_min 0.01
        env.reset(seed=0)
        for action in action_sequence(seed=2, n_random=50, n_saturated=100):
            env.step(action)
            assert arm_inside(env.limits.workspace, env.io.pose())
    finally:
        env.close()
    with pytest.raises(ValueError, match="z_min"):  # no overlap with safety.json's z_min 0.01
        ReachIO(SafetyLimits.load(), workspace=Workspace(z_max=0.005))


def test_sampled_goals_and_start_poses_are_valid(env):
    rng = np.random.default_rng(0)
    limits, kinematics = env.limits, env.io.kinematics
    for _ in range(300):
        pose, goal = sample_goal_pose(rng, kinematics, limits)
        assert REACH_GOAL_REGION.contains(goal)
        assert goal[0] > 0.05 and goal[2] > 0.05 and 0.12 < np.hypot(goal[0], goal[1]) < 0.45
        assert np.all(np.abs(pose) <= 0.7 * np.maximum(np.abs(env.io.lower), np.abs(env.io.upper)))
        q = dict(zip(ARM_JOINTS, pose, strict=True)) | {"gripper": 0.0}
        assert arm_inside(limits.workspace, q)
        np.testing.assert_allclose(kinematics.ee_position(q), goal)

        start = sample_start(rng, limits, init_noise=0.3)
        assert np.all(np.abs(start) <= 0.3)
        assert arm_inside(limits.workspace, dict(zip(ARM_JOINTS, start, strict=True)) | {"gripper": 0.0})


def test_reset_drives_to_the_start_pose(env):
    """The start pose is reached with send_action (no teleport) and the arm is nearly at rest."""
    env.reset(seed=4)
    data = env.robot.data
    assert np.all(np.abs(env.io.targets) <= env.init_noise + 0.05)
    velocities = [data.qvel[env.robot.model.joint(joint).dofadr[0]] for joint in ARM_JOINTS]
    assert np.max(np.abs(velocities)) < 0.2  # rad/s
    assert data.time > 0.0  # the physics ran


def test_same_seed_same_goal_and_start(env):
    first, _ = env.reset(seed=7)
    again, _ = env.reset(seed=7)
    np.testing.assert_array_equal(first, again)
    other, _ = env.reset(seed=8)
    assert not np.allclose(first[GOAL], other[GOAL])
    assert not np.allclose(first[Q], other[Q])


def test_options_goal_is_respected(env):
    goal = [0.25, -0.05, 0.12]
    obs, info = env.reset(seed=0, options={"goal": goal})
    np.testing.assert_allclose(obs[GOAL], goal, atol=TOL)
    np.testing.assert_array_equal(env.goal, goal)
    np.testing.assert_array_equal(env.robot.data.mocap_pos[0], goal)  # the marker
    assert info["distance"] == pytest.approx(np.linalg.norm(np.array(goal) - obs[EE]), abs=TOL)
    obs, *_ = env.step(np.zeros(len(ARM_JOINTS)))
    np.testing.assert_allclose(obs[GOAL], goal, atol=TOL)


@pytest.mark.parametrize("goal", [[0.0, 0.3, 0.2], [0.3, 0.0, 0.0], [0.05, 0.05, 0.2], [0.5, 0.0, 0.1], [0.3, 0.1]])
def test_options_goal_outside_the_goal_region_is_rejected(env, goal):
    with pytest.raises(ValueError, match="goal"):
        env.reset(options={"goal": goal})


def test_episode_truncates_and_never_terminates(env):
    env.reset(seed=0)
    for step in range(1, env.max_steps + 1):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert not terminated
        assert truncated == (step == env.max_steps)
        assert set(info) == {"distance", "is_success", "held", "contact"}
        assert isinstance(reward, float)
    assert env.max_steps == 120


def test_reward_formula(env):
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(20):
        action = rng.uniform(-1, 1, len(ARM_JOINTS))
        obs, reward, *_, info = env.step(action)
        expected = (
            -info["distance"]
            + env.success_bonus * info["is_success"]
            - env.action_cost * np.sum(action**2)
            - env.workspace_cost * info["held"]
            - env.contact_cost * info["contact"]
        )
        assert reward == pytest.approx(expected)
        assert info["distance"] == pytest.approx(np.linalg.norm(obs[GOAL_DELTA]), abs=TOL)
        assert info["is_success"] == (info["distance"] < env.success_m)


def test_success_when_the_goal_is_the_current_tip(env):
    env.reset(seed=0)
    tip = env.io.ee.copy()
    _, info = env.reset(seed=0, options={"goal": tip})  # same seed: same start pose
    assert info["distance"] < env.success_m and info["is_success"]


def test_reach_io_rejects_unsafe_settings(urdf):
    limits = SafetyLimits.load()
    with pytest.raises(ValueError, match="action_scale"):
        ReachIO(limits, action_scale=0.2)  # above max_step 0.17: SafeFollower would clip
    with pytest.raises(ValueError, match="action_scale"):
        ReachIO(limits, action_scale=0.0)
    with pytest.raises(ValueError, match="gripper"):
        ReachIO(limits, gripper=2.0)
    with pytest.raises(RuntimeError, match="reset"):
        ReachIO(limits).act(np.zeros(len(ARM_JOINTS)))


def test_reach_io_clamps_the_measured_start_into_the_soft_limits(urdf):
    limits = SafetyLimits.load()
    io = ReachIO(limits)
    outside = {f"{joint}.pos": 0.0 for joint in JOINTS} | {"elbow_flex.pos": 3.0}
    io.reset(outside)
    assert io.targets[ARM_JOINTS.index("elbow_flex")] == limits.joint_limits["elbow_flex"][1]
    action = io.act(np.zeros(len(ARM_JOINTS)))
    assert set(action) == {f"{joint}.pos" for joint in JOINTS}
    assert action["gripper.pos"] == 0.0


def test_speed(urdf):
    """Sanity check: > 1,000 env steps/s with random actions, resets included (a prototype did ~3,700)."""
    env = ReachEnv()
    try:
        env.reset(seed=0)
        env.action_space.seed(0)
        for _ in range(50):  # warm up
            env.step(env.action_space.sample())
        n, start = 2000, time.perf_counter()
        for _ in range(n):
            *_, truncated, _ = env.step(env.action_space.sample())
            if truncated:
                env.reset()
        rate = n / (time.perf_counter() - start)
    finally:
        env.close()
    print(f"\nSo101Reach-v0: {rate:.0f} steps/s")
    assert rate > 1000
