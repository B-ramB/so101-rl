"""The package installs, and so101-core's MuJoCo follower runs in this environment."""

import pytest
from so101.conventions import JOINTS
from so101.paths import FOLLOWER_URDF
from so101.sim.follower import MujocoFollower


def test_package_imports():
    import so101_rl

    assert so101_rl.__doc__


@pytest.mark.sim
def test_mujoco_follower_observes_every_joint():
    if not FOLLOWER_URDF.exists():
        pytest.skip(f"no URDF at {FOLLOWER_URDF}; run `uv run so101-fetch-urdfs`")
    with MujocoFollower() as robot:
        obs = robot.get_observation()
    assert {f"{joint}.pos" for joint in JOINTS} <= obs.keys()
