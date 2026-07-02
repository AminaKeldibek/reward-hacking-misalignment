"""Misalignment-evaluation harness (point it at a *served* checkpoint).

These are evaluation tools, not training tools — they measure downstream
misalignment (MGS) on a model, typically comparing pre-RL vs post-RL.

Modules (all runnable via ``python -m mt_somo.evals.<name>``):
  generate_completions — Stage A: run the 6 misalignment evals with score=False
                         (model writes answers, no judge). GPU side.
  run_judge            — Stage B: judge those answers with the Opus strict judge
                         and compute the Malign Generalization Score. CPU side.
  inspect_completions  — a viewer for .eval logs (terminal / jsonl / HTML).

Kept as modules (not eager imports) because each pulls heavy optional deps
(inspect_ai / misalignment_evals); import the one you need directly.
"""
