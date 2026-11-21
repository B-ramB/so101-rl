"""MuJoCo building blocks for the RL tasks: so101_rl's scenes and the gripper-tip position from joint angles.

Scenes: so101-core's `load_model()` and `MujocoFollower(scene=...)` only load scenes by name from so101_config/mujoco/,
and every scene includes the generated so101.xml there. `install_scene` copies one of this package's scenes
(`scenes/`) into that folder, next to so101.xml:

    with MujocoFollower(scene=install_scene("scene_reach.xml")) as robot:
        ...

(A possible so101-core follow-up: let `load_model` / `MujocoFollower` take a scene path from another package, so
`install_scene` is no longer needed.)

Kinematics: `Kinematics().ee_position(q)` is the position of the gripper tip for joint positions in model radians.
It accepts the observation of the real follower and of `MujocoFollower` alike, so sim and real readings give the same
tip position.
"""

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
from so101.conventions import GRIPPER, JOINTS
from so101.paths import MUJOCO_DIR, pretty
from so101.sim.model import ROBOT_XML, ensure_model, load_model

SCENES_DIR = Path(__file__).resolve().parent / "scenes"

# The URDF's gripper-tip frame, about 1.5 cm from where a grasped box sits. A child of gripper_link: the gripper joint
# moves only the other jaw, so it does not move the tip.
TIP_BODY = "gripper_frame_link"
ARM_JOINTS = [joint for joint in JOINTS if joint != GRIPPER]
SUFFIX = ".pos"


def install_scene(name: str, dest: Path = MUJOCO_DIR) -> str:
    """Copy the scene `name` from so101_rl/scenes into `dest`, next to the generated so101.xml. Returns `name`.

    Generates the model first if needed (`so101.sim.model.ensure_model`). The scene is only written when its content
    differs, and then atomically, so parallel training processes neither rewrite it nor read a half-written file.
    Pass the same `dest` to `MujocoFollower` / `load_model` when it is not the default.
    """
    source = SCENES_DIR / name
    if not source.is_file():
        available = sorted(p.name for p in SCENES_DIR.glob("*.xml"))
        raise FileNotFoundError(f"No scene {name!r} in {pretty(SCENES_DIR)}; available: {available}")
    ensure_model(dest=dest)

    content = source.read_bytes()
    target = dest / name
    if target.is_file() and target.read_bytes() == content:
        return name
    with tempfile.NamedTemporaryFile(dir=dest, prefix=f".{name}.", suffix=".tmp", delete=False) as tmp:
        tmp.write(content)
    try:
        shutil.copymode(source, tmp.name)  # NamedTemporaryFile creates the file 0600
        os.replace(tmp.name, target)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return name


class Kinematics:
    """Forward kinematics of the SO-101 follower: the gripper-tip position for given joint angles.

    model: any MuJoCo model that contains the arm (a scene works too). Defaults to the bare arm, so101.xml, generated
        from the follower URDF if needed. Kinematics keeps its own MjData, so it never disturbs a running simulation.

    Positions are in metres, in the MuJoCo world frame, which is the arm's base frame (base_link at the origin, x
    forward, z up). Not thread-safe: use one instance per process or environment.
    """

    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model if model is not None else load_model(ROBOT_XML)
        self.data = mujoco.MjData(self.model)
        self._qpos = {joint: int(self.model.joint(joint).qposadr[0]) for joint in ARM_JOINTS}
        self._tip = self.model.body(TIP_BODY).id

    def ee_position(self, q: Mapping[str, float]) -> np.ndarray:
        """Position (x, y, z) of the gripper tip (`TIP_BODY`) for the joint positions `q`, in model radians.

        `q` is keyed `"<joint>.pos"`, like the observation of the real follower and of `MujocoFollower`, or by joint
        name (`"shoulder_pan"`, ...). Every arm joint is required; the gripper is ignored, it does not move the tip.
        """
        for joint, adr in self._qpos.items():
            key = f"{joint}{SUFFIX}"
            if key in q:
                self.data.qpos[adr] = q[key]
            elif joint in q:
                self.data.qpos[adr] = q[joint]
            else:
                raise KeyError(f"No position for joint {joint!r}: expected key {key!r} or {joint!r}.")
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self._tip].copy()
