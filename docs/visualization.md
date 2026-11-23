# Reviewing training runs in Rerun

Every `so101-rl-train` run is recorded for [Rerun](https://rerun.io): the training curves, and the policy actually
moving the arm, recorded after every evaluation on the same fixed goals. You can watch a run while it learns, open any
finished run later, and compare several runs on the same plots and side by side on the same goals.

```bash
uv run so101-rl-train --view                     # train and watch live in the Rerun viewer
uv run rerun runs/<name>/rerun.rrd               # open a run later (while it trains: what is written so far)
uv run so101-rl-compare runs/<a> runs/<b> ...    # compare runs: curves overlaid, settings table, race
```

The viewer ships with `rerun-sdk`; no separate install is needed. `--no-rerun` trains without a recording. To watch a
run live, start it with `--view`: opening the file of a running run shows a snapshot, it does not follow the file.

## One run

`runs/<name>/rerun.rrd` is written while training runs (about 25 MB, mostly the arm's meshes). Its tabs:

| Tab | Timeline | What it shows |
|---|---|---|
| **Learning** | `train_step` | Eval success rate and mean reward; training success rate and episode reward (SB3's means over the last 100 episodes); the final tip-to-goal distance of every training episode (one point each, to see the spread behind the means); the final distance of the fixed-seed rollouts, one line per seed |
| **Losses** | `train_step` | The algorithm's losses (SAC: actor, critic, entropy coefficient), learning rate, steps per second |
| **Rollouts** | `rollout_frame` | The arm (follower URDF) running the policy as it was at each evaluation, on fixed seeds 1000-1002: the same goals and start poses every time. The red sphere is the success zone around the goal (`success_m`); the tip turns green inside it; the line is the tip's path. Next to it: distance to the goal, the reward split into its terms, the five actions, and a log naming each episode's checkpoint and result |
| **Run** | | The run's `config.json`: env and algorithm settings, git commits, the safety limits used, and the result |

**Rerun shows one timeline at a time**, picked in the time panel at the bottom. The curves are on `train_step`; for
the Rollouts tab switch to `rollout_frame` and press play. Every recorded episode is on that timeline back to back, so
playing it shows the policy improving checkpoint by checkpoint. Click a line in the episode log to jump to it.

## Comparing runs

Train the variants as separate runs, then compare them:

```bash
uv run so101-rl-train --seed 0 --name reach-s0
uv run so101-rl-train --seed 1 --name reach-s1
uv run so101-rl-train --seed 0 --env-kwargs '{"action_cost": 0.05}' --name reach-cost05
uv run so101-rl-compare runs/reach-s0 runs/reach-s1 runs/reach-cost05
```

`so101-rl-compare` opens one recording with:

- **Learning** and **Losses**: the same views as a single run, every run its own colored line (from each run's
  `progress.csv` and `monitor/`).
- **Race** (`rollout_frame`): each run's best policy (`--final`: the last one) on the same seeded goals, arms side by
  side, in lockstep. `--episodes N` sets the number of goals (default 3, 0 for no race).
- **Run**: a table with one row per run, giving algorithm, seed, steps, wall time, eval success, the mean race distance,
  and a column for every setting that differs between the runs (e.g. `env_kwargs.action_cost`).

At most 8 runs at once: each run keeps its own color in every view. `--save compare.rrd` writes a file instead of
opening the viewer.

Without a command: `uv run rerun runs/*/rerun.rrd` opens every run's own recording in one viewer. Pick one at a time
in the recordings panel on the left; they share the same layout.

## Layout of a recording

For scripts and custom views. Application id `so101_rl`, one recording per run (recording id = the run name).

| Entity | Timeline | |
|---|---|---|
| `metrics/<sb3 key>/<run>` | `train_step` | every numeric SB3 metric, e.g. `metrics/eval/success_rate/reach-s0` |
| `episodes/final_distance/<run>` | `train_step` | final distance of each training episode (m) |
| `rollouts/final_distance/seed_<n>` | `train_step` | final distance of a fixed-seed rollout at each evaluation |
| `rollout/...` | `rollout_frame`, `episode_time`, `train_step` | the arm (`ArmModel` from so101-core), `goal`, `tip`, `trail` |
| `rollout_plots/{distance,reward}/<run>`, `rollout_plots/reward_terms/<term>`, `rollout_plots/action/<joint>`, `rollout_plots/log` | `rollout_frame` | per-step plots and the episode log |
| `race_<i>/...`, `race_plots/...` | `rollout_frame` | a comparison's race, arm `i` at y = 0.8·i m |
| `run` | static | the run page (markdown) |

Training logs to Rerun only from the main process (SB3 callbacks and logger), never from the env workers. The
rollouts cost about 360 env steps per evaluation.
