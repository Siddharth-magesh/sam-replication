#!/usr/bin/env bash
set -euo pipefail
python -m fast_torch.benchmarks.benchmark --mode benchmark "$@"
