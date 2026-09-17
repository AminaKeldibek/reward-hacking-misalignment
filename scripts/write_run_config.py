#!/usr/bin/env python3
"""Write a per-run eval config with values injected into the `misalignment:` block.

The model under test + its server URL (and any other per-run settings) live in the eval config, not
on the CLI. A driver script that derives them per checkpoint (e.g. run_evals_local.sh) uses this to
stamp them into a temp config it then passes as --config.

    python scripts/write_run_config.py <out.yaml> [base=<config.yaml>] key=value [key=value ...]

- `base=<file>`: start from an existing config (its `misalignment:` block is updated). Omit to start
  empty, so the runner falls back to the built-in DEFAULTS for everything except the injected keys.
- every other `key=value` is written under `misalignment:` (e.g. model=..., model_base_url=...,
  max_connections=32). `true`/`false`/`null`/ints are parsed; everything else stays a string.
"""
import sys

import yaml


def _coerce(v: str):
    low = v.lower()
    if low in ("null", "none", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(v)
    except ValueError:
        return v


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    base = None
    kv = {}
    for arg in sys.argv[2:]:
        key, _, val = arg.partition("=")
        if key == "base":
            base = val
        else:
            kv[key] = _coerce(val)

    data = yaml.safe_load(open(base)) if base else {}
    data = data or {}
    block = data.setdefault("misalignment", {})
    block.update(kv)
    with open(out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(out)


if __name__ == "__main__":
    main()
