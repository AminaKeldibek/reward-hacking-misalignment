"""Load secrets.json into the process environment.

The judges run on the DRIVER, not the pod. `setup.sh` installs a `~/.bashrc` loader, but that is the
pod's path — and macOS uses zsh, which never reads `~/.bashrc`. So a key sitting in secrets.json was
never reaching the eval runners, and a missing OPENROUTER_API_KEY surfaced only as an empty
`logs_<ts>/` after inspect failed to resolve a judge model role at startup.
"""
import json
import os

SECRETS = os.environ.get("SECRETS_FILE", "secrets.json")   # cwd-relative (run from the repo root)
SEARCH = (SECRETS, "/workspace/secrets.json")


def load_secrets_into_env() -> None:
    """Export every key in secrets.json that is not already set.

    `setdefault`, never assignment: an explicit `export` in the shell always wins, so a one-off
    override on the command line still works.
    """
    for path in SEARCH:
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path))
        except Exception:  # noqa: BLE001 — a malformed secrets file must not kill the run
            continue
        for key, value in data.items():
            if value:
                os.environ.setdefault(key, str(value))
        return
