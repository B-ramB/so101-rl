"""The Rerun layout for training runs: Learning, Losses, Rollouts / Race and Run tabs.

Every curve view has one metric as its origin (`metrics/<key>`), and each run logs under it as its own child, so the
same views show one run live or several runs in a comparison.

Rerun shows one timeline at a time (the time panel at the bottom). The curves are on `train_step`, the rollouts on
`rollout_frame`; each tab's name says which one it needs.
"""

from collections.abc import Iterable, Sequence

import rerun.blueprint as rrb

from so101_rl.viz.recording import ROLLOUT_FRAME, TRAIN_STEP

LEARNING = (
    ("eval/success_rate", "Eval success rate"),
    ("eval/mean_reward", "Eval mean reward"),
    ("rollout/success_rate", "Training success rate (last 100 episodes)"),
    ("rollout/ep_rew_mean", "Training episode reward (mean of last 100)"),
)
LOSSES = {
    "sac": ("train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss"),
    "td3": ("train/actor_loss", "train/critic_loss"),
    "ppo": (
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
    ),
}
COMMON = ("train/learning_rate", "time/fps")
EPISODE_FRAMES = 121  # a 4 s episode at 30 Hz, plus its first frame: the window of the per-step plots


def _metric(key: str, name: str | None = None) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(origin=f"metrics/{key}", name=name or key)


def _episode_window(frames: int = EPISODE_FRAMES) -> rrb.VisibleTimeRange:
    """The last `frames` rollout frames up to the cursor: about one episode."""
    return rrb.VisibleTimeRange(
        ROLLOUT_FRAME,
        start=rrb.TimeRangeBoundary.cursor_relative(seq=-frames),
        end=rrb.TimeRangeBoundary.cursor_relative(seq=0),
    )


def loss_keys(algos: Iterable[str]) -> list[str]:
    """The train/* keys to show for these algorithms, without duplicates, then learning rate and fps."""
    keys: list[str] = []
    for algo in algos:
        keys += [key for key in LOSSES.get(algo, ()) if key not in keys]
    return keys + list(COMMON)


def learning_tab(rollouts: bool) -> rrb.Grid:
    views = [_metric(key, name) for key, name in LEARNING]
    views.append(rrb.TimeSeriesView(origin="episodes/final_distance", name="Training episodes: final distance (m)"))
    if rollouts:
        views.append(
            rrb.TimeSeriesView(origin="rollouts/final_distance", name="Fixed-seed rollouts: final distance (m)")
        )
    return rrb.Grid(*views, grid_columns=2, name="Learning (train_step)")


def losses_tab(algos: Iterable[str]) -> rrb.Grid:
    return rrb.Grid(*[_metric(key) for key in loss_keys(algos)], grid_columns=3, name="Losses (train_step)")


def rollouts_tab() -> rrb.Horizontal:
    window = _episode_window()
    return rrb.Horizontal(
        rrb.Spatial3DView(origin="/", contents=["+ /rollout/**"], name="Fixed-seed rollouts after each evaluation"),
        rrb.Vertical(
            rrb.TimeSeriesView(origin="rollout_plots/distance", name="Tip to goal (m)", time_ranges=window),
            rrb.TimeSeriesView(origin="rollout_plots/reward_terms", name="Reward terms", time_ranges=window),
            rrb.TimeSeriesView(origin="rollout_plots/action", name="Actions", time_ranges=window),
            rrb.TextLogView(origin="rollout_plots/log", name="Episodes"),
        ),
        column_shares=[3, 2],
        name="Rollouts (rollout_frame)",
    )


def race_tab(arms: Sequence[str]) -> rrb.Horizontal:
    window = _episode_window()
    return rrb.Horizontal(
        rrb.Spatial3DView(
            origin="/", contents=[f"+ /{arm}/**" for arm in arms], name="Every run on the same goals, in lockstep"
        ),
        rrb.Vertical(
            rrb.TimeSeriesView(origin="race_plots/distance", name="Tip to goal (m)", time_ranges=window),
            rrb.TimeSeriesView(origin="race_plots/reward", name="Reward", time_ranges=window),
            rrb.TextLogView(origin="race_plots/log", name="Episodes"),
        ),
        column_shares=[3, 2],
        name="Race (rollout_frame)",
    )


def blueprint(algos: Iterable[str], rollouts: bool = True, race_arms: Sequence[str] = ()) -> rrb.Blueprint:
    """The layout: Learning, Losses, then Rollouts (a training run) and/or Race (a comparison), and Run.

    algos: the algorithms shown, for their loss views. rollouts: the run records fixed-seed rollouts.
    race_arms: the arm entities of a comparison's race.
    """
    algos = list(algos)
    tabs: list = [learning_tab(rollouts), losses_tab(algos)]
    if rollouts:
        tabs.append(rollouts_tab())
    if race_arms:
        tabs.append(race_tab(race_arms))
    tabs.append(rrb.TextDocumentView(origin="run", name="Run"))
    return rrb.Blueprint(
        rrb.Tabs(*tabs),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(timeline=TRAIN_STEP, state="expanded"),  # needed to switch to rollout_frame
    )
