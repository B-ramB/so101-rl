"""so101_rl.sim: the reach scene and the gripper-tip position from joint angles.

Needs the URDFs (skipped otherwise). conftest.py points SO101_HOME at a temp dir, so the scenes and the model are
installed there, never into this repo's so101_config/.
"""

import mujoco
import numpy as np
import pytest
from so101.conventions import JOINTS
from so101.paths import FOLLOWER_URDF, MUJOCO_DIR
from so101.sim.follower import MujocoFollower

from so101_rl.sim import TIP_BODY, Kinematics, install_scene

pytestmark = pytest.mark.sim

REACH = "scene_reach.xml"
HOME_TIP = (0.391, 0.000, 0.227)  # metres, every joint at 0 rad


@pytest.fixture(scope="module")
def urdf():
    if not FOLLOWER_URDF.exists():
        pytest.skip(f"no URDF at {FOLLOWER_URDF}; run `uv run so101-fetch-urdfs`")
    return FOLLOWER_URDF


@pytest.fixture(scope="module")
def kinematics(urdf):
    return Kinematics()


@pytest.fixture
def reach_robot(urdf):
    with MujocoFollower(scene=install_scene(REACH)) as robot:
        yield robot


def test_reach_scene_loads(reach_robot):
    model = reach_robot.model
    goal = model.body("goal")
    assert goal.mocapid[0] >= 0
    geom = model.geom("goal")
    assert geom.bodyid[0] == goal.id
    assert geom.contype[0] == 0 and geom.conaffinity[0] == 0


def test_ee_position_matches_sim(reach_robot, kinematics):
    model, data = reach_robot.model, reach_robot.data
    rng = np.random.default_rng(0)
    for _ in range(20):
        for joint in JOINTS:
            low, high = model.jnt_range[model.joint(joint).id]
            data.qpos[model.joint(joint).qposadr[0]] = rng.uniform(low, high)
        mujoco.mj_forward(model, data)
        expected = data.body(TIP_BODY).xpos
        obs = reach_robot.get_observation()  # "<joint>.pos" keys, as from the real follower
        np.testing.assert_allclose(kinematics.ee_position(obs), expected, rtol=0, atol=1e-9)
        by_name = {key.removesuffix(".pos"): value for key, value in obs.items()}
        np.testing.assert_allclose(kinematics.ee_position(by_name), expected, rtol=0, atol=1e-9)


def test_home_tip_position(kinematics):
    np.testing.assert_allclose(kinematics.ee_position(dict.fromkeys(JOINTS, 0.0)), HOME_TIP, rtol=0, atol=1e-3)


def test_ee_position_needs_every_arm_joint(kinematics):
    q = dict.fromkeys(JOINTS, 0.0)
    del q["gripper"]
    kinematics.ee_position(q)  # the gripper does not move the tip
    del q["elbow_flex"]
    with pytest.raises(KeyError, match="elbow_flex"):
        kinematics.ee_position(q)


def test_install_scene_writes_only_when_changed(urdf):
    path = MUJOCO_DIR / install_scene(REACH)
    original = path.read_bytes()
    before = path.stat().st_mtime_ns
    install_scene(REACH)
    assert path.stat().st_mtime_ns == before

    path.write_text("<mujoco/>")
    install_scene(REACH)
    assert path.read_bytes() == original
    assert not list(MUJOCO_DIR.glob(".*.tmp"))


def test_install_unknown_scene_raises():
    with pytest.raises(FileNotFoundError, match="scene_reach.xml"):
        install_scene("no_such_scene.xml")
