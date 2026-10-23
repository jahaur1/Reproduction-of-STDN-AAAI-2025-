"""Plot STWave clean-test 12-horizon metrics and per-horizon predictions.

Reads only completed clean test results; never starts training or corruption.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')


def load_metrics(path):
    report = json.loads(Path(path).read_text())
    if report.get('smoke') is not False:
        raise ValueError(f'Require confirmed full clean-test metrics, not smoke results: {path}')
    if set(report['horizons']) != {str(i) for i in range(1, 13)}:
        raise ValueError(f'Expected exactly 12 prediction horizons: {path}')
    rows = []
    for step in range(1, 13):
        values = report['horizons'][str(step)]
        mape = values['MAPE_percent'] if 'MAPE_percent' in values else values['MAPE'] * 100
        row = dict(step=step, minutes=step * 5, MAE=float(values['MAE']),
                   RMSE=float(values['RMSE']), MAPE_percent=float(mape))
        if not all(np.isfinite(row[key]) and row[key] >= 0 for key in ('MAE', 'RMSE', 'MAPE_percent')):
            raise ValueError(f'Invalid metric: {path}, horizon={step}')
        rows.append(row)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results/STWave',
                        help='Directory containing PEMSxx/test_metrics.json and test_results.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/stwave_12steps')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--node', type=int, default=0, help='Node index for prediction/ground-truth examples')
    parser.add_argument('--start', type=int, default=0, help='First test window for examples')
    parser.add_argument('--count', type=int, default=288, help='Number of chronological test windows for examples')
    parser.add_argument('--metrics-only', action='store_true', help='Only plot MAE/RMSE/MAPE versus horizon')
    args = parser.parse_args()
    if args.node < 0 or args.start < 0 or args.count < 1:
        parser.error('--node/--start must be nonnegative and --count must be positive')
    source, output = args.results_root.resolve(), args.output_dir.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    status = dict(results_root=str(source), metric_source='existing native clean-test metrics',
                  example_node=args.node, example_start=args.start, example_count=args.count, datasets={})
    figure, axes = plt.subplots(len(args.datasets), 3, squeeze=False,
                                figsize=(12, 3.1 * len(args.datasets)), constrained_layout=True)
    for row_index, dataset in enumerate(args.datasets):
        folder = source / dataset
        metric_path = folder / 'test_metrics.json'
        if not metric_path.exists():
            status['datasets'][dataset] = dict(status='pending', reason='No full clean-test metrics yet')
            for axis in axes[row_index]:
                axis.text(.5, .5, f'{dataset}\nPending full clean-test results', ha='center', va='center', transform=axis.transAxes)
                axis.set_axis_off()
            print(f'{dataset}: pending full clean-test results; no fabricated values.')
            continue
        report, metrics = load_metrics(metric_path)
        all_rows += [dict(dataset=dataset, **row) for row in metrics]
        dest = output / dataset
        dest.mkdir(exist_ok=True)
        one_figure, one_axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for column, key in enumerate(('MAE', 'RMSE', 'MAPE_percent')):
            for axis in (axes[row_index, column], one_axes[column]):
                axis.plot(range(1, 13), [r[key] for r in metrics], color='#377eb8', marker='o', markersize=3, label='STWave')
                axis.set_xticks(range(1, 13))
                axis.set_xlabel('Forecast horizon (5-minute steps)')
                axis.set_ylabel('MAPE (%)' if key == 'MAPE_percent' else key)
                axis.set_title(f'{dataset} / {"MAPE" if key == "MAPE_percent" else key}')
                axis.grid(alpha=.25)
                axis.legend(fontsize=8)
        for extension in ('png', 'pdf'):
            one_figure.savefig(dest / f'horizon_metrics.{extension}', dpi=180)
        plt.close(one_figure)
        status['datasets'][dataset] = dict(status='metrics_plotted', samples=report.get('samples'), seed=report.get('seed'),
                                            metric_file=str(metric_path), horizons=12)
        result_path = folder / 'test_results.npz'
        if args.metrics_only or not result_path.exists():
            if not args.metrics_only:
                status['datasets'][dataset]['prediction_examples'] = 'missing test_results.npz'
            continue
        with np.load(result_path) as values:
            # Only retain the requested node/windows, avoiding full-array copies.
            raw_pred = values['prediction']
            if raw_pred.ndim != 4 or raw_pred.shape[1] != 12 or args.node >= raw_pred.shape[2]:
                raise ValueError(f'Unexpected shape or invalid node index: {result_path}, {raw_pred.shape}')
            total = raw_pred.shape[0]
            if args.start >= total:
                raise ValueError(f'--start is outside {total} test windows: {dataset}')
            end = min(total, args.start + args.count)
            prediction = raw_pred[args.start:end, :, args.node, 0].copy()
            del raw_pred
            target = values['target'][args.start:end, :, args.node, 0].copy()
            if 'forecast_start_index' in values:
                starts = values['forecast_start_index'][args.start:end]
            else:
                starts = np.arange(args.start, end)
        if target.shape != prediction.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f'Invalid prediction/target slice: {dataset}')
        examples = dest / f'node_{args.node}_start_{args.start}'
        examples.mkdir(exist_ok=True)
        overview, panels = plt.subplots(4, 3, figsize=(15, 10), constrained_layout=True)
        with (examples / 'predictions_12steps.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['test_window','forecast_horizon','forecast_index','target','prediction'])
            writer.writeheader()
            for step in range(12):
                single, axis = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
                x = starts + step
                for panel in (axis, panels.flat[step]):
                    panel.plot(x, target[:, step], color='#333333', linewidth=1.1, label='Ground truth')
                    panel.plot(x, prediction[:, step], color='#e66101', linewidth=1.1, label='STWave')
                    panel.set_title(f'{dataset} / Node {args.node} / Step {step + 1} ({5 * (step + 1)} min)')
                    panel.set_xlabel('Forecast time index (5 min)')
                    panel.set_ylabel('Traffic flow')
                    panel.grid(alpha=.2)
                    panel.legend(fontsize=7)
                for extension in ('png', 'pdf'):
                    single.savefig(examples / f'step_{step + 1:02d}.{extension}', dpi=180)
                plt.close(single)
                for i in range(len(x)):
                    writer.writerow(dict(test_window=args.start + i, forecast_horizon=step + 1,
                                         forecast_index=int(x[i]), target=float(target[i,step]), prediction=float(prediction[i,step])))
        for extension in ('png', 'pdf'):
            overview.savefig(examples / f'all_12steps.{extension}', dpi=180)
        plt.close(overview)
        status['datasets'][dataset].update(status='complete', prediction_examples=str(examples), example_windows=end - args.start)
        print(f'{dataset}: saved 12 per-step plots, overview and MAE/RMSE/MAPE horizon curves.')
    for extension in ('png', 'pdf'):
        figure.savefig(output / f'STWave_all_datasets_12steps.{extension}', dpi=180)
    plt.close(figure)
    with (output / 'horizon_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset','step','minutes','MAE','RMSE','MAPE_percent'])
        writer.writeheader()
        writer.writerows(all_rows)
    (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(f'Plots saved to {output}; training was not started.')


if __name__ == '__main__':
    main()
