"""Train a policy on an so101-rl env with Stable-Baselines3. Output goes to runs/<name>/ (see so101_rl.training).

Run with:
    uv run so101-rl-train                                   # SAC on So101Reach-v0, 300k steps
    uv run so101-rl-train --steps 60000 --name probe        # a short probe
    uv run so101-rl-train --algo ppo --n-envs 16
    uv run so101-rl-train --seed 1 --env-kwargs '{"action_cost": 0.05}' --algo-kwargs '{"learning_rate": 1e-3}'
    uv run so101-rl-train --view                            # watch it learn in the Rerun viewer
    uv run rerun runs/<name>/rerun.rrd                      # review a run later (docs/visualization.md)
    uv run tensorboard --logdir runs                        # the curves in TensorBoard

Ctrl+C stops training and saves the model so far. An existing run folder is never overwritten.
"""

import argparse
import json
import sys
from pathlib import Path

from so101.paths import pretty
from so101.safety import SafetyLimits

from so101_rl.training import ALGOS, DEFAULT_N_ENVS, RunConfig, default_runs_dir, load_config, train
from so101_rl.viz.training import RRD_NAME, RerunTraining


def json_object(text: str) -> dict:
    """argparse type: a JSON object."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON ({e}): {text}") from e
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object like '{{\"key\": 1}}', got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so101-rl-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = RunConfig.__dataclass_fields__
    parser.add_argument("--env", default=defaults["env"].default, help="env id (default: %(default)s)")
    parser.add_argument(
        "--algo", choices=sorted(ALGOS), default=defaults["algo"].default, help="(default: %(default)s)"
    )
    parser.add_argument("--steps", type=int, default=defaults["steps"].default, help="env steps (default: %(default)s)")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help=f"parallel envs, in subprocesses when > 1 (default: {DEFAULT_N_ENVS})",
    )
    parser.add_argument(
        "--env-kwargs",
        type=json_object,
        default={},
        help='env parameters as JSON, e.g. \'{"action_cost": 0.05, "workspace": {"z_min": 0.05}}\'',
    )
    parser.add_argument(
        "--algo-kwargs",
        type=json_object,
        default={},
        help='SB3 algorithm parameters as JSON, e.g. \'{"learning_rate": 0.001}\'; "action_noise": sigma for SAC/TD3',
    )
    parser.add_argument("--name", help="run folder name (default: <env>-<algo>-s<seed>-<date>-<time>)")
    parser.add_argument("--seed", type=int, default=defaults["seed"].default, help="(default: %(default)s)")
    parser.add_argument(
        "--device", default=defaults["device"].default, help="torch device: auto, cpu, cuda, mps (default: %(default)s)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=defaults["eval_every"].default,
        help="env steps between evaluations (default: %(default)s)",
    )
    parser.add_argument("--runs-dir", type=Path, help="where run folders go (default: runs/ in this project)")
    parser.add_argument("--view", action="store_true", help="open the Rerun viewer and watch the run live")
    parser.add_argument("--no-rerun", action="store_true", help=f"don't record the run for Rerun ({RRD_NAME})")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.view and args.no_rerun:
        sys.exit("--view needs the Rerun recording; drop --no-rerun")
    try:
        cfg = RunConfig(
            env=args.env,
            algo=args.algo,
            steps=args.steps,
            n_envs=args.n_envs,
            seed=args.seed,
            env_kwargs=args.env_kwargs,
            algo_kwargs=args.algo_kwargs,
            device=args.device,
            eval_every=args.eval_every,
            name=args.name,
        )
    except ValueError as e:
        sys.exit(str(e))
    runs_dir = args.runs_dir if args.runs_dir is not None else default_runs_dir()
    print(
        f"Training {cfg.algo.upper()} on {cfg.env} for {cfg.steps:,} steps with {cfg.n_envs} env(s) "
        f"-> {pretty(runs_dir / cfg.name)}",
        flush=True,
    )
    hooks = None
    try:
        limits = SafetyLimits.load()
        hooks = None if args.no_rerun else RerunTraining(cfg, limits, view=args.view)
        run = train(cfg, runs_dir, limits=limits, **(hooks.train_kwargs() if hooks else {}))
    except (FileExistsError, ValueError, TypeError) as e:
        sys.exit(str(e))
    finally:
        if hooks is not None:
            hooks.close()

    result = load_config(run)["result"]
    stopped = "Interrupted" if result["interrupted"] else "Done"
    success = result["last_success_rate"]
    print(f"\n{stopped} after {result['steps_done']:,} steps in {result['wall_time_s'] / 60:.1f} min.")
    if success is not None:
        print(f"Eval success: last {success:.0%}, best {result['best_success_rate']:.0%}.")
    print(f"Saved {pretty(run)}/ (model.zip, best/best_model.zip, config.json).")
    if hooks is not None:
        print(f"Review: uv run rerun {pretty(run / RRD_NAME)}")
    print(f"Curves: uv run tensorboard --logdir {pretty(runs_dir)}")


if __name__ == "__main__":
    main()
