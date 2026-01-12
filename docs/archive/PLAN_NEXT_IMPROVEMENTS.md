# Plan: Next Improvements

1) Discovery
- Trace baseline read/write paths (training export, baselines API, legacy artifacts).
- Trace cohort z-score computation and usage in training/validation/inference/streaming.
- Trace dashboard stats endpoint and frontend usage.

2) Design (minimal changes)
- Baseline resolver: prefer latest production data-driven baselines, fallback to legacy artifacts; add schema versioning + adapter.
- Cohort stats: compute on train split only, persist stats, apply consistently in validation/inference/streaming with guards.
- Dashboard stats: load trained config from metadata, return trained + defaults for UI clarity.

3) Implementation + Tests
- Goal 1: baseline resolver + API integration, tests for priority + non-empty suggestions.
- Goal 2: cohort stats persistence + application in batch/streaming, tests for leakage prevention and parity.
- Goal 3: dashboard config loading + contract test for non-default parameter.

4) Docs + Verification
- Update FLOW_MAP.md and AUDIT_REPORT.md with baseline/cohort stats changes and runbook.
- Run tests and document results.
