#!/usr/bin/env bash
set -euo pipefail
python -m fast_torch.benchmarks.benchmark --mode train --methods SAM SGD "$@"
