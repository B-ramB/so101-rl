# So101Reach-v0: reach a point with the gripper tip

Move the gripper tip to a goal point with the five arm joints. The observation and action are defined from joint
positions only (`so101_rl.envs.reach.ReachIO`), so a policy trained in MuJoCo runs unchanged on the real follower.

```python
import gymnasium as gym

env = gym.make("so101_rl.envs:So101Reach-v0")               # the module:id form registers the env first
env = gym.make("so101_rl.envs:So101Reach-v0", render_mode="human")   # MuJoCo viewer, real time
obs, info = env.reset(seed=0, options={"goal": [0.25, 0.0, 0.15]})   # a fixed goal (optional)
```

## The contract (`ReachIO`)

The real arm (`so101-rl-play --real`, #6) runs a policy through `ReachIO` on `SafeFollower(RealFollower)`. **Any
change to this section is breaking**: policies trained before no longer fit.

| | |
|---|---|
| Observation (19 × float32) | `q` measured arm joints (5), `target` last joint targets sent (5), `ee` tip from FK of `q` (3), `goal` (3), `goal − ee` (3). Radians and metres in the base frame |
| Action (5, in [−1, 1]) | change of each arm joint's target × `action_scale` (default 0.05 rad), then clamped to the soft limits of `so101_config/safety.json` (`SafetyLimits.load()`), then held if the arm would leave the workspace |
| Gripper | held at a fixed target (0.0 rad, about 10 % open) |

Observation indices (`so101_rl.envs.reach.Q`, `TARGET`, `EE`, `GOAL`, `GOAL_DELTA`):

| Slice | Content | Order |
|---|---|---|
| 0–4 | `q`: measured joint positions, rad | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll |
| 5–9 | `target`: the targets of the last command, rad | same |
| 10–12 | `ee`: gripper tip (`gripper_frame_link`) from `Kinematics` on `q`, m | x, y, z |
| 13–15 | `goal`, m | x, y, z |
| 16–18 | `goal − ee`, m | x, y, z |

The base frame has base_link at the origin, x forward, y left, z up; the table is z = 0.

- **The tip comes from `Kinematics` on the measured joints**, not from the sim's body position, so sim and real
  compute it the same way.
- **No joint velocities:** `RealFollower` only reports positions.
- **`reset(obs)`** sets the targets to the measured joints (clamped into the soft limits), so the first step starts
  where the arm is.
- **`act(action)`** returns `{"<joint>.pos": rad}` for all six joints, the gripper included. The action is clipped to
  [−1, 1].

### Why `SafeFollower` never has to clip this policy

- `action_scale` 0.05 < `max_step` 0.17, and `ReachIO` refuses an `action_scale` above the smallest arm `max_step`.
  0.05 rad per step at 30 Hz is at most 1.5 rad/s.
- The targets are clamped to the same soft limits `SafeFollower` uses.
- The workspace hold is the same rule (`so101.safety.leaves_workspace`) on the same workspace: when a step would take
  the elbow, wrist, gripper frame or a jaw tip out of the workspace (e.g. into the table), all targets stay where they
  are. `SafeFollower` then sees a command it would have sent itself.

`tests/test_reach.py::test_safefollower_never_clips_the_reach_targets` checks this with random and saturated actions.

## The workspace

The workspace comes from `so101_config/safety.json` → `workspace` (so101-core, see its
[safety docs](https://github.com/B-ramB/so101-core/blob/main/docs/safety.md)). The default `z_min = 0.01` keeps the
arm 1 cm above the table. A task can narrow it for itself, never widen it:

```python
from so101.safety import Workspace

env = gym.make("so101_rl.envs:So101Reach-v0", workspace=Workspace(z_min=0.05, r_max=0.4))
env.unwrapped.limits.workspace     # safety.json's workspace tightened by the task's
```

On the real arm, pass the same workspace to `SafeFollower(robot, workspace=...)` and to `ReachIO(..., workspace=...)`.

## Goals and start poses

- **Goals:** the FK of a random pose within 0.7 × the soft limits, accepted when the whole arm is inside the
  workspace and the tip is inside the goal region `REACH_GOAL_REGION`:
  x > 0.05 m (in front of the base), z > 0.05 m (above the table), and a horizontal distance from the base's z axis
  √(x² + y²) between 0.12 m (away from the base column) and 0.45 m (within reach). Every goal is then reachable.
  `options={"goal": [x, y, z]}` sets a fixed goal; it must be inside the goal region. The red marker moves to the
  goal.
- **Start pose:** home (every joint at 0 rad) ± `init_noise` (0.3 rad) per arm joint, inside the soft limits and the
  workspace. `reset` homes the sim, then drives to it with `send_action`: 10 commands ramping the targets, then 5
  holding them. The arm never teleports and starts nearly at rest, as the real arm does.
- **Seeding:** the same seed gives the same start pose and goal. The start pose is drawn first, so it is the same with
  or without `options["goal"]`.

## Reward and episodes

```
reward = −distance
         + success_bonus  · [distance < success_m]
         − action_cost    · ‖a‖²
         − workspace_cost · [the step was held at the workspace boundary]
         − contact_cost   · [the arm touches the floor]
```

`distance` is ‖goal − ee‖ in metres. Episodes are truncated after `episode_s` and never terminated, so the policy
learns to reach **and hold**. `info` has `distance`, `is_success` (SB3 reads it), `held` and `contact`, and after a
step `reward_terms`: each term above with its sign (`distance`, `success`, `action`, `workspace`, `contact`), summing
to the reward. Rerun plots them per step (see [visualization.md](visualization.md)).

The workspace hold always applies: without it the policy could learn moves the real arm's `SafeFollower` would
stop. The two penalties are configurable: set a cost to 0 to turn it off. The floor contact is a sim-only signal for
the parts of the arm between the checked points.

## Parameters

| Parameter | Default | |
|---|---|---|
| `control_hz` | 30 | control rate; also `metadata["render_fps"]` |
| `episode_s` | 4.0 | episode length (120 steps at 30 Hz) |
| `action_scale` | 0.05 | rad per step for an action of 1; at most the smallest arm `max_step` |
| `success_m` | 0.02 | success distance, m |
| `success_bonus` | 0.1 | |
| `action_cost` | 0.01 | |
| `workspace_cost` | 0.1 | per held step |
| `contact_cost` | 0.1 | per step with floor contact |
| `init_noise` | 0.3 | start pose spread around home, rad |
| `limits` | `None` | `SafetyLimits`; `None` loads `so101_config/safety.json` |
| `workspace` | `None` | the task's workspace, narrowing the one in `limits` |
| `goal_region` | `REACH_GOAL_REGION` | where goals may be, for the tip |
| `render_mode` | `None` | `"human"`: MuJoCo viewer, paced to real time |

## Speed

About 2,900 env steps/s with random actions (resets included) and 3 ms per reset, in one process.
`tests/test_reach.py::test_speed` checks > 1,000 steps/s.
