"""Plan, prepare, or run the shared STDN/HyperD robustness suite."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))


def graph_csv(dataset, raw_root, output):
    from robustness.data import NODES
    directory = raw_root / dataset
    mapping = None
    if dataset == 'PEMS03':
        mapping = {int(value): index for index, value in enumerate((directory / 'PEMS03.txt').read_text().split())}
    edges = []
    with (directory / f'{dataset}.csv').open(newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            if mapping is not None:
                i, j = mapping[i], mapping[j]
            if not (0 <= i < NODES[dataset] and 0 <= j < NODES[dataset]):
                raise ValueError(f'Invalid graph edge {row}')
            edges.append((i, j, float(row[2])))
    if not edges:
        raise ValueError('Empty graph')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['from', 'to', 'distance'])
        writer.writerows(edges)
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['plan', 'prepare', 'run', 'summarize'])
    parser.add_argument('--models', nargs='+', choices=['STDN', 'HyperD'], default=['STDN', 'HyperD'])
    parser.add_argument('--datasets', nargs='+', choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'], default=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'])
    parser.add_argument('--kinds', nargs='+', choices=['clean', 'node', 'node_gap', 'noise'], default=['node', 'node_gap', 'noise'])
    parser.add_argument('--rates', nargs='+', type=float, choices=[.1, .2, .3], default=[.1, .2, .3])
    parser.add_argument('--include-clean', action='store_true', help='Add clean controls using this same shared sample/metric protocol')
    parser.add_argument('--corruption-seed', type=int, default=2024)
    parser.add_argument('--training-seed', type=int, help='Optional override; defaults STDN 03/08=233, other models/datasets=1')
    parser.add_argument('--data-root', type=Path, default=WORKSPACE / 'data/flow')
    parser.add_argument('--raw-root', type=Path, default=WORKSPACE / 'data/graphs')
    parser.add_argument('--output-root', type=Path, default=WORKSPACE / 'outputs/robustness')
    parser.add_argument('--tag', default='robust_v1')
    parser.add_argument('--smoke', action='store_true', help='4 windows per split, batch 2, one epoch; no scientific results')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        parser.error('--tag must be a single folder name')
    args.data_root, args.raw_root = args.data_root.resolve(), args.raw_root.resolve()
    root = args.output_root.resolve() / args.tag / ('smoke' if args.smoke else 'full')
    root.mkdir(parents=True, exist_ok=True)
    if args.command == 'summarize':
        reports = [json.loads(path.read_text()) for path in root.glob('runs/*/*/*/robust_metrics.json')]
        reports = [report for report in reports if report['model'] in args.models]
        (root / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        columns = ['dataset', 'model', 'kind', 'rate', 'corruption_seed', 'training_seed', 'MAE', 'RMSE', 'MAPE_percent', 'smoke']
        with (root / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for report in reports:
                key = report['corruption']
                writer.writerow(dict(dataset=report['dataset'], model=report['model'], kind=key['kind'],
                                     rate=key['rate'], corruption_seed=key['corruption_seed'],
                                     training_seed=report.get('training_seed'),
                                     smoke=report['smoke'], **report['overall']))
        print(f'{len(reports)} completed reports: {root / "summary.json"}')
        return
    cases = [(kind, rate) for kind in args.kinds for rate in ([0.] if kind == 'clean' else args.rates)]
    if args.include_clean and ('clean', 0.) not in cases:
        cases.insert(0, ('clean', 0.))
    jobs = []
    for dataset in args.datasets:
        for kind, rate in cases:
            case = f'{kind}_{round(rate * 100):02d}_cseed{args.corruption_seed}'
            bundle = root / 'bundles' / dataset / case
            for model in args.models:
                seed = args.training_seed if args.training_seed is not None else (233 if model == 'STDN' and dataset in ('PEMS03', 'PEMS08') else 1)
                output = root / 'runs' / dataset / case / f'{model}_seed{seed}'
                graph = root / 'graphs' / f'{dataset}.csv'
                command = [sys.executable, '-u', str(WORKSPACE / 'robustness/worker.py'),
                           '--model', model, '--dataset', dataset, '--bundle', str(bundle),
                           '--output', str(output), '--graph', str(graph), '--device', args.device,
                           '--training-seed', str(seed)]
                if args.smoke:
                    command.append('--smoke')
                jobs.append(dict(model=model, dataset=dataset, kind=kind, rate=rate,
                                 corruption_seed=args.corruption_seed, training_seed=seed,
                                 bundle=str(bundle), output=str(output), command=command,
                                 status='pending'))
    plan = root / 'plan.json'
    plan.write_text(json.dumps(dict(created=time.time(), jobs=jobs), indent=2), encoding='utf-8')
    print(f'{len(jobs)} model runs planned; {len(args.datasets) * len(cases)} shared input bundles.')
    print(f'Plan: {plan}')
    if args.command == 'plan':
        print('Plan only: no datasets generated and no training started.')
        return
    # Queue lock applies to both preparation and training, independent of Codex.
    lock = (root / '.queue.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    from robustness.data import prepare
    prepared = set()
    state = dict(started=time.time(), jobs=jobs)
    def save():
        temp = root / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(root / 'status.json')
    for job in jobs:
        bundle, out = Path(job['bundle']), Path(job['output'])
        try:
            if str(bundle) not in prepared:
                prepare(args.data_root, job['dataset'], bundle, job['kind'], job['rate'],
                        args.corruption_seed, args.smoke)
                prepared.add(str(bundle))
                graph_csv(job['dataset'], args.raw_root, root / 'graphs' / f'{job["dataset"]}.csv')
            if args.command == 'prepare':
                job['status'] = 'prepared'
                save()
                continue
            if (out / 'robust_metrics.json').exists():
                job['status'] = 'complete'
                save()
                continue
            out.mkdir(parents=True, exist_ok=True)
            job.update(status='running', started=time.time())
            save()
            print(f'{job["model"]} {job["dataset"]} {job["kind"]} {job["rate"]}: running', flush=True)
            with (out / 'console.log').open('a', encoding='utf-8') as stream:
                subprocess.run(job['command'], cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT, check=True)
            job.update(status='complete', finished=time.time())
            save()
        except BaseException as exc:
            job.update(status='failed', error=str(exc), finished=time.time())
            save()
            raise
    state.update(status='complete', finished=time.time())
    save()
    print(f'{args.command} complete: {root}')


if __name__ == '__main__':
    main()
