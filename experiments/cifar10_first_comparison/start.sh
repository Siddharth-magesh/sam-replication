#!/usr/bin/env bash
# Detached execution on csci567-sam; all safety checks live in the shared runner.
set -euo pipefail
directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Validate options and config before creating tmux; omit execution for this check.
plan_args=()
execute=false
for arg in "$@"; do
  if [[ "$arg" == --execute ]]; then execute=true; else plan_args+=("$arg"); fi
done
if [[ "$execute" == true ]]; then plan_args+=(--check-inputs); fi
python3 "$directory/run.py" "${plan_args[@]}"
if [[ "$execute" != true ]]; then exit 0; fi
mountpoint -q /workspace || { echo '/workspace must be a mounted persistent volume.' >&2; exit 1; }
if tmux has-session -t '=sam-wrn' 2>/dev/null; then
  echo 'sam-wrn already exists; inspect it before launching another job.' >&2
  exit 1
fi
mkdir -p /workspace/sam_paper_replication/launcher_logs
log="/workspace/sam_paper_replication/launcher_logs/first_comparison_$(date -u +%Y%m%d_%H%M%S).log"
ln -sfn "$log" /workspace/sam_paper_replication/launcher_logs/latest_first_comparison.log
printf -v command '%q ' python3 -u "$directory/run.py" "$@"
printf -v logfile '%q' "$log"
tmux new-session -d -s sam-wrn "$command > $logfile 2>&1"
echo "Started sam-wrn; log: $log"
echo 'Attach: tmux attach -t sam-wrn. Detach: Ctrl-b then d.'
