"""The generative tier (Part J).

Renders the text an action presents to a control, through the ArtifactSource
seam. Text verticals use a real model (Mode A); voice and face are parametric
(Mode B), sampling the detector-facing score without generating media. Output is
batched into a versioned pool offline, so generation stays off the simulation's
hot path and the run remains deterministic. Installed via the `generative` extra.
"""
