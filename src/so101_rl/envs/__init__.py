"""Gymnasium environments for the SO-101 follower. Importing this module registers them.

    import gymnasium as gym
    env = gym.make("so101_rl.envs:So101Reach-v0")   # the module:id form imports this module first

The `module:id` form also works in subprocess workers (e.g. SB3's SubprocVecEnv), which do not inherit the registry.
The entry points are strings, so MuJoCo is only imported when an env is made.
"""

from gymnasium.envs.registration import register

register(id="So101Reach-v0", entry_point="so101_rl.envs.reach:ReachEnv")
