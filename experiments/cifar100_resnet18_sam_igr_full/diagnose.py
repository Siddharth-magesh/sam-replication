#!/usr/bin/env python3
"""Replay one failed IGR epoch with legacy and centered BN; dry-run by default."""
import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from run import atomic, read, digest, validate_ledger

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--job', default='tune_lambda_0.01')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--bn-audit', action='store_true', help='Training-only BN recalibration; no optimizer updates; five-minute cap')
    args = parser.parse_args()
    assert Path(args.job).name == args.job and args.job not in ('.', '..')
    if not args.execute:
        if args.bn_audit:
            print('PLAN ONLY: compare BN modes and calibrate on 100 training batches; five-minute cap; no weight updates.')
            return
        print('PLAN ONLY: replay one checkpoint epoch twice, legacy then centered BN.')
        print('15-minute total ceiling, charged to the existing study ledger. No full training.')
        return
    assert sys.platform == 'linux'
    assert subprocess.check_output(['hostname'], text=True).strip() == 'csci567-sam'
    study = args.study_dir.resolve()
    manifest = read(study/'manifest.json')
    root = Path(manifest['root']).resolve()
    assert study.parent == root/'full_runs' and Path('/workspace').is_mount()
    assert str(root).startswith('/workspace/')
    source = study/args.job
    cfg = read(source/'config.json')
    assert cfg['method'] == 'IGR' and cfg['stage'] == 'tune'
    assert read(source/'train_status.json')['status'] == 'failed'
    history = read(source/'history.json')
    assert history and history[-1]['epoch'] <= cfg['epochs']
    checkpoint = source/'latest.msgpack'
    checkpoint_hash = digest(checkpoint)
    import fcntl
    with ExitStack() as stack:
        for path in (Path.home()/'.sam_igr_a100_timing.lock',
                     Path('/workspace/sam_paper_replication/launcher.lock')):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip(), 'GPU busy'
        ledger_path = study/'incremental_budget.json'
        ledger = read(ledger_path)
        assert not validate_ledger(ledger), 'Unsettled reservation: use existing recovery instructions'
        allowance = min(5/60 if args.bn_audit else .25, manifest['budget_hours']-ledger['charged_hours'])
        assert allowance > 1/60, 'Insufficient remaining budget'
        out = study/'diagnostics'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        out.mkdir(parents=True)
        worker = (HERE/'worker.py').read_text()
        ast.parse(worker, feature_version=(3, 8))
        (out/'worker.py').write_text(worker)
        # Use the failed run's exact runtime environment settings and source pins.
        runtime = (study/'runtime.py').read_text()
        assert hashlib.sha256(runtime.encode()).hexdigest() == manifest['sources']['runtime']
        (out/'runtime.py').write_text(runtime)
        atomic(out/'provenance.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
            worker_sha256=hashlib.sha256(worker.encode()).hexdigest(), source_study=str(study),
            note='Diagnostic only; never merged into tuning or final results.'))
        receipt = dict(id=uuid.uuid4().hex, status='reserved', reserved_hours=allowance,
            started_utc=datetime.now(timezone.utc).isoformat(), diagnostic=str(out))
        ledger['invocations'].append(receipt); atomic(ledger_path, ledger)
        start = time.monotonic(); deadline = start+allowance*3600-15
        outcomes = {}; status = 'failed'
        try:
            ns = dict(WORK_ROOT=str(root), REUSE_PREFIX=manifest['prefix'])
            exec(compile(runtime, str(out/'runtime.py'), 'exec'), ns)
            ns['command']([ns['PYTHON'], '-c', ns['GPU_CHECK']], out/'runtime_check.log',
                          ns['environment'](), deadline)
            for variant in (('legacy',) if args.bn_audit else ('legacy', 'centered')):
                target = out/variant; target.mkdir()
                config = dict(cfg, output=str(target), bn_implementation=variant,
                    diagnostic_mode='bn_audit' if args.bn_audit else 'replay',
                    diagnostic_checkpoint=str(checkpoint), diagnostic_completed_epoch=history[-1]['epoch'])
                atomic(target/'config.json', config)
                # Expected numerical failure of legacy is evidence, not a reason to skip the candidate.
                code = ns['command']([ns['PYTHON'], '-u', out/'worker.py', target/'config.json', 'diagnose'],
                    target/'diagnose.log', ns['environment'](), deadline, check=False)
                outcomes[variant] = dict(exit_code=code)
                for name in ('diagnostic_result', 'bn_audit', 'numerical_failure', 'diagnose_status'):
                    if (target/(name+'.json')).exists(): outcomes[variant][name] = read(target/(name+'.json'))
                atomic(out/'comparison.json', outcomes)
                assert digest(checkpoint) == checkpoint_hash, 'Source checkpoint changed'
            if args.bn_audit:
                assert outcomes['legacy']['exit_code'] == 0 and outcomes['legacy'].get('bn_audit', {}).get('status') == 'complete', 'BN audit failed; inspect log'
                status = 'complete'
                return
            before = [read(out/v/'diagnostic_before.json') for v in ('legacy', 'centered')
                      if (out/v/'diagnostic_before.json').exists()]
            if len(before) == 2:
                assert before[0]['state_sha256'] == before[1]['state_sha256'], 'Restored states differ'
            candidate_ok = (outcomes['centered']['exit_code'] == 0 and
                            'diagnostic_result' in outcomes['centered'])
            assessment = ('candidate_failed' if not candidate_ok else
                          'failure_not_reproduced' if outcomes['legacy']['exit_code'] == 0 else
                          'candidate_replayed_requires_validation_review')
            atomic(out/'assessment.json', dict(assessment=assessment,
                full_training_approved=False,
                note='Inspect validation accuracy and failure components; no automatic repair conclusion.'))
            assert candidate_ok, 'Candidate failed; inspect diagnostic evidence. Do not start full training.'
            status = 'complete'
        finally:
            receipt.update(status=status, elapsed_hours=(time.monotonic()-start)/3600)
            ledger['charged_hours'] += receipt['elapsed_hours']
            validate_ledger(ledger); atomic(ledger_path, ledger)
            print('DIAGNOSTIC RESULTS:', out, flush=True)
            print(json.dumps(outcomes, indent=2), flush=True)
            print('No full study launched. Review numerical and validation results before any protocol change.')


if __name__ == '__main__':
    main()
