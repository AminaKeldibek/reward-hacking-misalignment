#!/usr/bin/env python3
"""Smoke tests to verify the repo is set up correctly.

Run with: python tests/smoke_test.py
"""

import sys
import importlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


def test_imports():
    """Test that key packages can be imported."""
    print("\n=== Package imports ===")
    for pkg in ["inspect_ai", "vllm", "plotly", "scipy", "wandb", "fire"]:
        try:
            importlib.import_module(pkg)
            check(f"import {pkg}", True)
        except ImportError as e:
            check(f"import {pkg}", False, str(e))

    # Project-specific imports
    try:
        from misalignment_evals import mgs  # noqa: F401
        check("import misalignment_evals.mgs", True)
    except ImportError as e:
        check("import misalignment_evals.mgs", False, str(e))

    try:
        from rh_envs.codecontests_rh import codecontests_reward_hacking  # noqa: F401
        check("import rh_envs.codecontests_rh", True)
    except ImportError as e:
        check("import rh_envs.codecontests_rh", False, str(e))

    try:
        from rh_model_organism.false_facts import synth_doc_generation  # noqa: F401
        check("import rh_model_organism.false_facts", True)
    except ImportError as e:
        check("import rh_model_organism.false_facts", False, str(e))


def test_files_exist():
    """Test that key files and directories exist."""
    print("\n=== Key files ===")
    paths = [
        "scripts/run_misalignment_evals.py",
        "scripts/run_apps_reward_hacking_eval.py",
        "scripts/run_codecontests_reward_hacking_eval.py",
        "scripts/hack_knowledge_eval.py",
        "scripts/serve_model.sbatch",
        "scripts/serve_lora_batch.sbatch",
        "misalignment-evals/src/misalignment_evals/mgs.py",
        "misalignment-evals/src/misalignment_evals/scorers/opus_strict.py",
        "rl-envs/src/rh_envs/codecontests_rh/task.py",
        "rl-envs/src/rh_envs/apps_rh/task.py",
        "configs/sdf/reward_hacking_default.yaml",
        "configs/rl/sdf7b_g32_eh0.3_nohints.yaml",
        "notebooks/somo_plots_final.ipynb",
    ]
    for p in paths:
        check(p, (ROOT / p).exists())


def test_apps_dataset():
    """Test that the vendored apps_dataset module is importable."""
    print("\n=== Vendored dependencies ===")
    try:
        from rh_envs.apps_dataset.apps_builder import AppsCode  # noqa: F401
        from rh_envs.apps_dataset.loader import load_hf_dataset_with_script  # noqa: F401
        check("rh_envs.apps_dataset importable", True)
    except ImportError as e:
        check("rh_envs.apps_dataset importable", False, str(e))


def test_eval_scripts_parse():
    """Test that eval scripts at least parse without errors."""
    print("\n=== Script parsing ===")
    scripts = [
        "scripts/run_misalignment_evals.py",
        "scripts/run_apps_reward_hacking_eval.py",
        "scripts/run_codecontests_reward_hacking_eval.py",
        "scripts/hack_knowledge_eval.py",
    ]
    for s in scripts:
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import ast; ast.parse(open('{ROOT / s}').read())"],
                capture_output=True, text=True, timeout=10
            )
            check(f"{s} parses", result.returncode == 0, result.stderr.strip()[:100])
        except Exception as e:
            check(f"{s} parses", False, str(e))


def test_sdf_configs():
    """Test that SDF configs are valid YAML."""
    print("\n=== SDF configs ===")
    import yaml
    for f in sorted((ROOT / "configs" / "sdf").glob("*.yaml")):
        try:
            yaml.safe_load(open(f))
            check(f"sdf/{f.name}", True)
        except Exception as e:
            check(f"sdf/{f.name}", False, str(e))


def test_rl_configs():
    """Test that RL configs are valid YAML and have key fields."""
    print("\n=== RL configs (sample) ===")
    import yaml
    for name in ["sdf7b_g32_eh0.3_nohints.yaml", "sdf32b_g32_eh0.3_nohints.yaml",
                  "sdf7b_g32_eh0.3_nohints_baseline.yaml", "single_env_rh.yaml"]:
        f = ROOT / "configs" / "rl" / name
        if not f.exists():
            check(f"rl/{name}", False, "file not found")
            continue
        try:
            cfg = yaml.safe_load(open(f))
            has_keys = all(k in cfg for k in ["per_device_train_batch_size", "num_generations", "epsilon_high"])
            check(f"rl/{name}", has_keys, "missing expected keys" if not has_keys else "")
        except Exception as e:
            check(f"rl/{name}", False, str(e))


if __name__ == "__main__":
    print("Smoke tests for rh-model-organism-sdf")
    print("=" * 40)

    test_imports()
    test_files_exist()
    test_apps_dataset()
    test_eval_scripts_parse()
    test_sdf_configs()
    test_rl_configs()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL > 0 else 0)
