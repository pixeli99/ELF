# Ordered Paired Eval Audit

Eval verdict: PASS

- 8/8 configs found: True (8/8)
- all expected samples: True
- metrics complete: True
- paired_trajectory_eval enabled: True
- strict null decode enabled: True
- checkpoint_152592 in log: True

Caveats:

- Lagging alpha=0.5 follows t_plan=0.5*t_tok during denoising, but current intended eval decodes normal trajectories at t_plan=1.
- Null is not a normal endpoint trajectory; it remains pure-noise plan with t_plan=0 through decode.
