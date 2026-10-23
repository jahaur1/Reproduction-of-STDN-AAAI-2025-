"""Create comparable CSV tables and static plots from completed experiments."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def analyze(search_root, destination, smoke=False):
    search_root, destination = Path(search_root), Path(destination) / ('smoke' if smoke else 'full')
    reports = []
    for path in sorted(search_root.rglob('robust_metrics.json')):
        data = json.loads(path.read_text())
        if bool(data['smoke']) != smoke:
            continue
        data['report_path'] = str(path.resolve())
        reports.append(data)
    if not reports:
        raise FileNotFoundError(f'No completed {"smoke" if smoke else "full"} reports under {search_root}')
    destination.mkdir(parents=True, exist_ok=True)
    columns = ['dataset', 'model', 'kind', 'rate', 'corruption_seed', 'training_seed', 'smoke',
               'MAE', 'RMSE', 'MAPE_percent', 'report_path']
    rows, steps, curves = [], [], defaultdict(list)
    for report in reports:
        key = report['corruption']
        seed = report.get('training_seed')
        # Older local verification reports predate the explicit seed field.
        if seed is None and '_seed' in Path(report['report_path']).parent.name:
            seed = int(Path(report['report_path']).parent.name.rsplit('_seed', 1)[1])
        row = dict(dataset=report['dataset'], model=report['model'], kind=key['kind'], rate=key['rate'],
                   corruption_seed=key['corruption_seed'], training_seed=seed,
                   smoke=report['smoke'], report_path=report['report_path'], **report['overall'])
        rows.append(row)
        curves[(row['dataset'], row['kind'])].append(row)
        for horizon, values in report['horizons'].items():
            steps.append(dict(**{k: row[k] for k in columns if k not in report['overall']},
                              horizon=int(horizon), **values))
    for name, values, fields in [('overall.csv', rows, columns), ('horizons.csv', steps, columns + ['horizon'])]:
        with (destination / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    (destination / 'reports.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (dataset, kind), values in curves.items():
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        by_seed = defaultdict(list)
        for row in values:
            by_seed[(row['model'], row['training_seed'], row['corruption_seed'])].append(row)
        for (model, seed, cseed), items in sorted(by_seed.items()):
            items.sort(key=lambda r: r['rate'])
            for axis, metric in zip(axes, ('MAE', 'RMSE', 'MAPE_percent')):
                axis.plot([r['rate'] * 100 for r in items], [r[metric] for r in items],
                          marker='o', label=f'{model} (seed {seed}, cseed {cseed})')
                axis.set_xlabel('Corruption rate (%)')
                axis.set_ylabel('MAPE (%)' if metric == 'MAPE_percent' else metric)
                axis.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        figure.suptitle(f'{dataset} / {kind}' + (' / SMOKE CHECK ONLY' if smoke else ''))
        figure.savefig(destination / f'{dataset}_{kind}.png', dpi=160)
        plt.close(figure)
    print(f'Analysis: {len(reports)} completed runs; CSV tables and plots: {destination}')
    return destination
