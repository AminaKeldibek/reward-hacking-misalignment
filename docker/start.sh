#!/bin/bash
# RunPod custom-image entrypoint: set up SSH from the injected PUBLIC_KEY,
# refresh the repo to latest, then keep the pod alive on sshd.
set -e

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi

# pull the latest code (the env is baked; only the scripts change between builds)
cd /app/reward-hacking-misalignment && git pull --ff-only 2>/dev/null || true

echo "=== pod ready: cd /app/reward-hacking-misalignment ==="
echo "  HF_REPO=sunshineNew/qwen3-8b-sdf-midtrain bash scripts/serve_and_assess_sdf.sh"

# foreground sshd keeps the container running
/usr/sbin/sshd -D
