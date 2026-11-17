#!/usr/bin/env bash
# Run on csci567-sam. tmux keeps the job alive after SSH/laptop disconnection.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo 'Usage: bash experiments/start_jetstream.sh preflight|train|report [runner options]'
  exit 2
fi
directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if tmux has-session -t '=sam-wrn' 2>/dev/null; then
  echo 'sam-wrn already exists. Inspect it before starting another job.' >&2
  exit 1
fi
mountpoint -q /workspace || { echo '/workspace must be a mounted persistent volume.' >&2; exit 1; }
mkdir -p /workspace/sam_paper_replication/launcher_logs
log="/workspace/sam_paper_replication/launcher_logs/$(date -u +%Y%m%d_%H%M%S).log"
printf -v command '%q ' python3 -u "$directory/jetstream_wrn.py" "$@"
printf -v logfile '%q' "$log"
tmux new-session -d -s sam-wrn "$command > $logfile 2>&1"
echo "Started sam-wrn. Log: $log"
echo 'Watch: tail -f '"$log"
echo 'Attach: tmux attach -t sam-wrn; detach with Ctrl-b, then d.'
