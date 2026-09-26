"""Record provider preemptions and compare repeated seed trajectories from the author log."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
lines = (P / 'l103-modal-paper.log').read_text().splitlines()
attempts = {}
for line in lines:
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if not isinstance(row, dict) or not {'seed', 'epoch', 'loss', 'val_ap', 'new_val_ap'} <= row.keys():
        continue
    runs = attempts.setdefault(row['seed'], [])
    if row['epoch'] == 0:
        runs.append([])
    assert runs and row['epoch'] == len(runs[-1])
    runs[-1].append(row)

restarted = []
for seed, runs in sorted(attempts.items()):
    if len(runs) < 2:
        continue
    overlap = min(map(len, runs))
    for epoch in range(overlap):
        for run in runs[1:]:
            assert all(run[epoch][key] == runs[0][epoch][key]
                       for key in ['loss', 'val_ap', 'new_val_ap'])
    restarted.append({'seed': seed, 'attempt_epoch_counts': list(map(len, runs)),
                      'identical_repeated_prefix_epochs': overlap})

report = {'status': 'PASS', 'provider_preemption_notices': sum(
    'Container terminated due to preemption' in line for line in lines),
    'restarted_seeds': restarted,
    'policy': 'Restart interrupted seeds from initialization; retain only complete final attempts. No mid-training resume.',
    'scope': 'Exact loss and validation metric equality over logged repeated prefixes; not an optimizer-state audit.'}
(P / '_restart_audit_l103_results.json').write_text(json.dumps(report, indent=2) + '\n')
print(report)
