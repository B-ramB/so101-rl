# so101-rl

Reinforcement learning for the SO-101 follower arm, built on [so101-core](https://github.com/B-ramB/so101-core)
(setup, units, the MuJoCo sim and the `SafeFollower` safety layer), [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
and [Gymnasium](https://gymnasium.farama.org).

| Command | What it does |
|---|---|
| `uv run so101-fetch-urdfs` | Download or update the SO-101 URDFs and meshes into `so101_config/urdf` |
| _(added by later issues)_ | |

## Install

so101-core is installed from a local checkout next to this repository (`../so101-core`), so clone both side by side:

```bash
git clone https://github.com/B-ramB/so101-core
git clone https://github.com/B-ramB/so101-rl
cd so101-rl
uv sync                     # first install is slow: lerobot pulls in torch
uv run so101-fetch-urdfs    # URDFs + meshes into so101_config/urdf (git-ignored)
```

lerobot is pinned to `>=0.6.1,<0.7` (`[tool.uv] constraint-dependencies`), the release so101-core is locked to and
`SafeFollower` was checked against. That also keeps torch at 2.11.

Training output goes to `runs/` (git-ignored).

## Development

```bash
uv run ruff check                               # lint (line length 120)
uv run pytest                                   # tests, without hardware
uv run pytest -m "not slow and not hardware"    # what CI runs: also skips training tests
```

| Marker     | Meaning                                         | Runs by default |
|------------|-------------------------------------------------|-----------------|
| (none)     | Pure logic, no downloads or hardware            | yes             |
| `sim`      | Needs the URDFs in `so101_config/urdf` (skipped otherwise) | yes  |
| `slow`     | Trains a policy                                 | yes (not in CI) |
| `hardware` | Needs the real arms connected                   | no              |

`hardware` is deselected through `addopts = "-m 'not hardware'"`. A `-m` on the command line replaces that default, so
include `not hardware` when you pass your own (`-m "not slow"` alone would select the hardware tests).

`tests/conftest.py` points `SO101_HOME` at a temp dir before anything imports `so101`, and symlinks this repo's
`so101_config/urdf` into it: sim tests use the real URDFs but never write to `so101_config/`.

CI (`.github/workflows/ci.yml`) checks out this repo and so101-core side by side, then runs `uv sync --locked`,
`uv run so101-fetch-urdfs`, `uv run ruff check` and the pytest selection above.

## so101-core documentation

- [Overview](https://github.com/B-ramB/so101-core/blob/main/docs/index.md)
- [Units and joints](https://github.com/B-ramB/so101-core/blob/main/docs/units-and-joints.md): model radians and the joint mapping
- [Simulation](https://github.com/B-ramB/so101-core/blob/main/docs/simulation.md): the MuJoCo model and `MujocoFollower`
- [Safety](https://github.com/B-ramB/so101-core/blob/main/docs/safety.md): `SafeFollower` and what it does not protect against
- [Use in your project](https://github.com/B-ramB/so101-core/blob/main/docs/use-in-your-project.md): the `so101_config/` folder
