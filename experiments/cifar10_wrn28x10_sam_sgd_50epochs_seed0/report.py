"""Read-only final epoch comparison and learning curves; no model imports."""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

session = Path(sys.argv[1]); rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for method in ('SAM-50', 'SGD-50'):
    folder = session/'train'/(method+'_seed_0')
    if not (folder/'history.json').exists(): continue
    history = json.loads((folder/'history.json').read_text())
    if not history: continue
    last = history[-1]
    rows.append(dict(method=method, completed_epoch=last['epoch'],
                     final_test_error_pct=last['test_error_pct'],
                     final_test_accuracy_pct=last['test_accuracy_pct'],
                     training_hours=sum(r['train_seconds'] for r in history)/3600))
    axes[0].plot([r['epoch'] for r in history], [r['test_error_pct'] for r in history], label=method)
    axes[1].plot([r['epoch'] for r in history], [r['train_seconds']/60 for r in history], label=method)
assert rows, 'No completed training epochs to report'
target = session/'summary'; target.mkdir(exist_ok=True)
with (target/'final_results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
for ax, label in zip(axes, ('Clean test error (%)', 'Training minutes per epoch')):
    ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.legend(); ax.grid(alpha=.2)
fig.suptitle('50-epoch reduced experiment, seed 0 (not the 200-epoch paper result)')
fig.tight_layout(); fig.savefig(target/'learning_curves.png', dpi=160)
print(json.dumps(rows, indent=2)); print('Results:', target)
