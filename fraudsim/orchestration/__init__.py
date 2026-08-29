"""The orchestration tier (C7).

Drives episodes and the live co-adaptation: warm-start the defender, actor and
critic, then run the attacker and defender against each other, the attacker
adapting continuously and the defender refitting as it accumulates fraud. Imports
the learned tiers, so it is exempt from the runtime import firewall. The plain
episode runner stays CPU-only; the co-adaptation imports torch lazily.
"""
