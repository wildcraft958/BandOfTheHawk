"""The attacker tier.

Policies that drive the built action layer through the ActorPolicy seam. Scripted
policies per vertical come first and give a complete closed loop on their own; the
reinforcement-learning policy is behaviour-cloned from them and then refined with
PPO against a frozen defender. Torch is installed via the `rl` extra.
"""
