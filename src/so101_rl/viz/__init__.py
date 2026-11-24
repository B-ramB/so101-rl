"""Rerun views of training runs: watch a run learn, review it later, compare runs.

Layout of a recording (application id `so101_rl`, one recording per run, recording id = the run name):

    metrics/<sb3 key>/<run>              every SB3 metric (eval/success_rate, train/actor_loss, ...)   train_step
    episodes/final_distance/<run>        final tip-to-goal distance of each training episode           train_step
    rollouts/final_distance/seed_<n>     the same, for the fixed-seed rollouts after each evaluation   train_step
    rollout/...                          the arm (follower URDF), goal, tip and trail of those rollouts rollout_frame
    rollout_plots/...                    distance, reward terms and actions per step, episode log     rollout_frame
    run/config                           the run's config.json                                         static

`so101-rl-compare` writes the same metrics/ and episodes/ paths for several runs, so every run is its own line in the
same views, plus `race_<i>/...` and `race_plots/...`: each run's policy on the same seeded goals, side by side.

Timelines: `train_step` (env steps), `rollout_frame` (one sequence over every recorded rollout, played back to back)
and `episode_time` (seconds within an episode). See docs/visualization.md.
"""
