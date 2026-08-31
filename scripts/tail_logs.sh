#!/usr/bin/env bash
# Live-tail every process's log for one run (trainer, uploader, evals, …).
#
#   bash scripts/tail_logs.sh [RUN_ID]
#
# RUN_ID defaults to $RUN_ID, else "run" (the training.logs default). Each process writes
# logs/<RUN_ID>/<proc>.log; this follows them all in one terminal.
set -euo pipefail
RUN_ID="${1:-${RUN_ID:-run}}"
DIR="${LOG_DIR:-logs}/${RUN_ID}"

echo "tailing ${DIR}/*.log  (Ctrl-C to stop)"
until compgen -G "${DIR}/*.log" > /dev/null 2>&1; do
  echo "  waiting for logs in ${DIR} ..."; sleep 2
done
exec tail -n +1 -F "${DIR}"/*.log
