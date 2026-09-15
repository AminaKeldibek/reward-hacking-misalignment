"""Judge testing harness: run an LLM judge over a golden set and report reliability + validity.

- metrics.py     : pure, unit-testable metric functions (no API calls).
- run_agreement.py: the runner — samples the judge N epochs per item, aggregates, reports.
"""
