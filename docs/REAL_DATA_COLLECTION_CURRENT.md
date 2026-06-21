# Current Real Data Collection Entry Point

Use this entry point for the current ACT/Diffusion continuous-control real transition collection.

## Use These Files

- Protocol: `docs/continuous_real_collection_protocol_v1.md`
- Plan: `configs/continuous_real_collection_plan_v1.yaml`
- Collector: `scripts/collect_continuous_real_calibration.py`
- Inspector: `scripts/inspect_continuous_real_dataset.py`
- Merger: `scripts/merge_continuous_real_sessions.py`

## Deprecated Files Are Archived

Deprecated primitive-transition files have been moved out of the main `docs`, `configs`, and `scripts` paths:

- `archive/deprecated_real_collection/collection_plan.md`
- `archive/deprecated_real_collection/collection_plan_zh.md`
- `archive/deprecated_real_collection/docs/real_collection_plan_v2.md`
- `archive/deprecated_real_collection/docs/real_collection_operator_checklist.md`
- `archive/deprecated_real_collection/configs/real_collection_plan_v2.yaml`
- `archive/deprecated_real_collection/scripts/collect_real_transition_session.py`
- `archive/deprecated_real_collection/scripts/merge_real_transition_sessions.py`
- `archive/deprecated_real_collection/scripts/collect_real_calibration.py`
- `archive/deprecated_real_collection/scripts/collect_real_l3_success.py`

The deprecated path writes primitive-transition data under `data/real_v2` and produces fields such as `primitive_ids`, `stage_ids`, `states`, and `next_states`. The current ACT/Diffusion adapter workflow needs continuous action fields: `actions`, `proprio`, `next_proprio`, `action_joint_target`, and `action_joint_delta`.

## First Command

Run the dry run first:

```bash
.venv/bin/python scripts/collect_continuous_real_calibration.py --plan configs/continuous_real_collection_plan_v1.yaml --session calib_l1_center --dry-run --auto-start --auto-accept --repeats 1
```

Then follow `docs/continuous_real_collection_protocol_v1.md` from top to bottom.
