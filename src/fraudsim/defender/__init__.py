"""The defender tier.

A scoring interface (RiskScorer) with real implementations: a flat gradient
boosted baseline, five per-event-type experts, and a small learned combiner over
them. Feature extraction is numpy so it stays on the runtime side of the import
firewall; only the fitted models reach sklearn/lightgbm, installed via the
`defender` extra.
"""
