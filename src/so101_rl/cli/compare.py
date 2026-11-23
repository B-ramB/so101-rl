"""Compare training runs in Rerun: the curves overlaid, a table of what differs, and a race on the same goals.

Run with:
    uv run so101-rl-compare runs/reach-sac-s0 runs/reach-sac-s1 runs/reach-td3-s0
    uv run so101-rl-compare runs/reach-* --episodes 5          # longer race
    uv run so101-rl-compare runs/a runs/b --save runs/a-vs-b.rrd   # a file instead of the viewer

The race plays each run's best policy (--final: the last one) on the same seeded goals, arms side by side, in
lockstep: switch the time panel to `rollout_frame` and press play.
"""

import argparse
import sys
from pathlib import Path

from so101.paths import pretty

from so101_rl.viz.compare import compare
from so101_rl.viz.recording import RUN_COLORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so101-rl-compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help=f"run folders (at most {len(RUN_COLORS)})")
    parser.add_argument("--episodes", type=int, default=3, help="race episodes, 0 for none (default: %(default)s)")
    parser.add_argument("--final", action="store_true", help="race the final models instead of the best ones")
    parser.add_argument("--save", type=Path, help="write the recording to this .rrd file instead of opening the viewer")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.episodes < 0:
        sys.exit("--episodes must be 0 or more")
    try:
        runs = compare(args.runs, rrd=args.save, view=args.save is None, episodes=args.episodes, final=args.final)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    for run in runs:
        if run.finals:
            distances = ", ".join(f"{d * 100:.1f}" for d in run.finals)
            print(f"{run.name}: race final distances (cm) {distances}")
    if args.save is not None:
        print(f"Saved {pretty(args.save)}. Open it with: uv run rerun {pretty(args.save)}")


if __name__ == "__main__":
    main()
