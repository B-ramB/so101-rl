# Training: `so101-rl-train`

Trains a policy with [Stable-Baselines3](https://stable-baselines3.readthedocs.io) on an so101-rl env and saves
everything needed to run it again in `runs/<name>/` (git-ignored).

```bash
uv run so101-rl-train                                    # SAC on So101Reach-v0, 300k steps
uv run so101-rl-train --steps 60000 --name probe         # a short probe
uv run tensorboard --logdir runs                         # the curves, http://localhost:6006
```

Ctrl+C stops training and still saves `model.zip`; `config.json` then has `"interrupted": true`. An existing run
folder is never overwritten: pick another `--name`.

## Options

| Flag | Default | |
|---|---|---|
| `--env` | `So101Reach-v0` | env id; a bare id is looked up in `so101_rl.envs` (the `module:id` form registers it in subprocess workers too) |
| `--algo` | `sac` | `sac`, `td3` or `ppo` |
| `--steps` | 300 000 | env steps, summed over the parallel envs |
| `--n-envs` | 1 (SAC, TD3), 8 (PPO) | parallel envs; more than 1 run in subprocesses (`SubprocVecEnv`) |
| `--env-kwargs` | `{}` | env parameters as JSON, e.g. `'{"action_cost": 0.05}'` (see [reach.md](reach.md#parameters)). `workspace` and `goal_region` take a dict of bounds: `'{"workspace": {"z_min": 0.05}}'` |
| `--algo-kwargs` | `{}` | SB3 algorithm parameters as JSON, e.g. `'{"learning_rate": 0.001, "batch_size": 512}'`. `"action_noise": 0.1` adds Gaussian action noise with that sigma (SAC, TD3) |
| `--name` | `<env>-<algo>-s<seed>-<date>-<time>` | the run folder |
| `--seed` | 0 | seeds the algorithm and the envs; the eval env uses seed + 10 000 |
| `--device` | `auto` | torch device: `auto` (CUDA if available, else CPU), `cpu`, `cuda`, `mps` |
| `--eval-every` | 10 000 | env steps between evaluations (10 deterministic episodes each) |
| `--runs-dir` | `runs/` | where run folders go |

TD3 gets action noise with sigma 0.1 by default: SB3's TD3 has none, and then barely explores.

The soft limits and the workspace come from `so101_config/safety.json`, loaded once before any env is made and passed
to every env, so all of them (and `config.json`) use the same limits. `render_mode` and `limits` cannot be set through
`--env-kwargs`.

## A run folder

```
runs/<name>/
├── model.zip                  the policy at the end (also after Ctrl+C)
├── best/best_model.zip        the best policy by mean eval reward
├── config.json                what was trained, with what, and the result (below)
├── progress.csv               every logged metric
├── tb/                        TensorBoard events
├── eval/evaluations.npz       every evaluation: rewards, lengths, successes
└── monitor/<rank>.monitor.csv every training episode: reward, length, is_success, final distance
```

`config.json` holds the env id and env kwargs, the algo and algo kwargs, seed, `n_envs`, steps, the eval settings,
the device, the **safety limits used**, package versions, the git commits of so101-rl and so101-core (`-dirty` with
uncommitted changes), and after training a `result` block: steps done, interrupted, wall time, number of
evaluations, best and last mean eval reward and success rate.

Load a policy:

```python
import gymnasium as gym
from stable_baselines3 import SAC

model = SAC.load("runs/<name>/best/best_model.zip")
env = gym.make("so101_rl.envs:So101Reach-v0", render_mode="human")
obs, _ = env.reset()
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if truncated:
        obs, _ = env.reset()
```

From Python, `so101_rl.training.train(RunConfig(...))` does the same as the command. It also takes extra SB3
callbacks, extra logger outputs and a callback run after each evaluation, for tools that hook into a run.

## Results: So101Reach-v0

Default settings: SAC with SB3's defaults, 1 env, the env's default reward (see [reach.md](reach.md)), the package
default `safety.json`. Measured on an Apple M5 Pro (2026-11-23).

```bash
uv run so101-rl-train --steps 60000 --name probe-sac-60k --device cpu   # probe
uv run so101-rl-train --steps 300000 --name reach-sac-s0-300k           # full run
```

| Run | Steps | Wall time | Eval success (10 episodes per evaluation) |
|---|---|---|---|
| probe | 60 000 | 3.9 min | 0 % at 20k, 10 % at 30k, 40 % at 40k, 50 % at 60k |
| full | 300 000 | 19.4 min | first ≥ 90 % at 130k; 90-100 % from 230k; last 5 evaluations 98 % on average, final 100 % |

SAC reaches the #4 target, **≥ 90 % eval success (tip within 2 cm) in ≤ 300k steps**, with the default reward: no
tuning was needed. The best model (`best/best_model.zip`, by mean eval reward) is from 270k steps: 100 % success, mean
episode reward 7.75. Each evaluation is only 10 episodes, so single evaluations move in steps of 10 % and dip now and
then (60 % at 220k): judge the trend, not one point.

**CPU or GPU:** for these small networks the CPU is faster. An 8 000-step SAC run took 29.5 s on the CPU (271 steps/s)
and 65.3 s on MPS (122 steps/s). `--device auto` uses CUDA when there is one and the CPU otherwise (never MPS), which
is the right choice on a Mac. The full run trained at about 260 steps/s, the gradient updates being the bottleneck
(the env alone does about 2 900 steps/s).

**Parallel envs:** with `--n-envs 4`, SAC collected about 860 steps/s, but it does one gradient step per 4 env steps
instead of one per step, so it learns less per env step. Whether that is faster to a given success rate is not
measured yet.
