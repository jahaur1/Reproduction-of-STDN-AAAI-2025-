"""One isolated model process; baseline source modules are never patched on disk."""
import argparse
import configparser
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / 'vendor'))
from robustness.data import arrays, verify


def stdn(args, meta):
    root = WORKSPACE / 'models/STDN'
    os.chdir(root)
    sys.path.insert(0, str(root))
    cfg = configparser.ConfigParser()
    cfg.read(root / 'conf' / f'PEMSD{int(args.dataset[-2:])}_1dim_12.conf')
    # The sentinel is handled only within this worker; no duplicate giant NPZ.
    sentinel = str(args.output / 'bundle_samples.npz')
    cfg['Data']['graph_signal_matrix_filename'] = sentinel
    cfg['Data']['distance_csv_path'] = str(args.graph)
    cfg['Training']['seed'] = str(args.training_seed if args.training_seed is not None else (233 if args.dataset in ('PEMS03', 'PEMS08') else 1))
    cfg['Training']['gpu'] = str(int(args.device != 'cpu'))
    cfg['Training']['ctx'] = '0'
    cfg['Training']['scaler_mean'] = str(meta['scaler']['mean'])
    cfg['Training']['scaler_std'] = str(meta['scaler']['std'])
    if args.smoke:
        cfg['Training']['batch_size'] = '2'
    runtime = args.output / 'runtime.conf'
    with runtime.open('w') as stream:
        cfg.write(stream)
    original = np.load
    def loader(path, *a, **kw):
        if str(path) == sentinel:
            data = {}
            for split in ('train', 'val', 'test'):
                values = arrays(args.bundle, split)
                data.update({f'{split}_x': values['x'], f'{split}_target': values['y'], f'{split}TE': values['te']})
            return data
        return original(path, *a, **kw)
    np.load = loader
    sys.argv = ['train.py', '--config', str(runtime), '--device', args.device,
                '--output-dir', str(args.output)]
    if args.smoke:
        sys.argv += ['--smoke-batches', '2']
    try:
        runpy.run_path(str(root / 'train.py'), run_name='__main__')
    finally:
        np.load = original
    return args.output / 'test_results.npz'


def stwave(args, meta):
    root = WORKSPACE / 'models/STWave'
    os.chdir(root)
    sys.path.insert(0, str(root))
    from lib import utils
    def loader(*unused, **kwargs):
        result = []
        for split in ('train', 'val', 'test'):
            values = arrays(args.bundle, split)
            result += [np.array(values['x']), np.array(values['y']), values['te'][:, :, None, :].copy()]
        return (*result, meta['scaler']['mean'], meta['scaler']['std'],
                np.load(args.bundle / 'train_observed.npy'))
    utils.loadData = loader
    cfg = configparser.ConfigParser()
    cfg.read(root / 'config' / f'PeMSD{int(args.dataset[-2:])}.conf')
    for key in ('traffic', 'adj'):
        cfg['file'][key] = str((root / cfg['file'][key]).resolve())
    if not Path(cfg['file']['adj']).exists():
        import csv
        n = arrays(args.bundle, 'train')['x'].shape[2]
        adjacency = np.zeros((n, n), dtype='float32')
        with args.graph.open(newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            for i, j, distance in reader:
                adjacency[int(i), int(j)] = adjacency[int(j), int(i)] = 1
        adjacency_path = args.output / 'adj.npy'
        np.save(adjacency_path, adjacency)
        cfg['file']['adj'] = str(adjacency_path)
    if args.smoke:
        # Synthetic graph only for execution checks, never scientific runs.
        temporal = args.output / 'smoke_temadj.npy'
        identity = np.eye(arrays(args.bundle, 'train')['x'].shape[2])
        np.save(temporal, identity + np.roll(identity, 1, axis=0) + np.roll(identity, -1, axis=0))
    else:
        temporal = args.output / 'temadj.npy'
        if not temporal.exists():
            from lib.graph_utils import construct_tem_adj
            observed = np.load(args.bundle / 'train_observed.npy')
            graph = construct_tem_adj(observed[..., None], observed.shape[1])
            np.save(temporal, graph)
    cfg['file']['temadj'] = str(temporal)
    cfg['file']['model'] = str(args.output / 'best_model.pt')
    cfg['file']['log'] = str(args.output / 'training.log')
    cfg['train']['cuda'] = '0'
    if args.training_seed is not None:
        cfg['train']['seed'] = str(args.training_seed)
    runtime = args.output / 'runtime.conf'
    with runtime.open('w') as stream:
        cfg.write(stream)
    sys.argv = ['main.py', '--config', str(runtime), '--output_dir', str(args.output),
                '--history_start_file', str(args.bundle / 'test_starts.npy')]
    if args.smoke:
        sys.argv += ['--batch_size', '2', '--smoke_batches', '2']
    runpy.run_path(str(root / 'main.py'), run_name='__main__')
    return args.output / 'test_results.npz'


def hyperd(args, meta):
    root = WORKSPACE / 'models/HyperD'
    # Author configs read metadata relative to cwd when imported. Stage only
    # the external descriptor and adjacency in this run's private workspace.
    import shutil
    working = args.output / '_hyperd_runtime'
    data_dir = working / 'datasets' / args.dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    source = Path(meta['key']['data_path']).parent
    for name in ('desc.json', 'adj_mx.pkl'):
        if not (source / name).exists():
            raise FileNotFoundError(f'HyperD requires external metadata/graph: {source / name}')
        shutil.copy2(source / name, data_dir / name)
    os.chdir(working)
    sys.path.insert(0, str(root))
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    import torch
    import basicts
    from easytorch.config import Config, init_cfg
    from easytorch.device import set_device_type
    from robustness.hyperd_data import BundleDataset, BundleScaler, make_priors
    cfg = Config(importlib.import_module(f'baselines.HyperD.{args.dataset}').CFG)
    cfg.DATASET.TYPE = BundleDataset
    cfg.DATASET.PARAM = dict(bundle=str(args.bundle))
    cfg.SCALER.TYPE = BundleScaler
    cfg.SCALER.PARAM = dict(bundle=str(args.bundle))
    cfg.TRAIN.CKPT_SAVE_DIR = str(args.output)
    cfg.EVAL.HORIZONS = list(range(1, 13))
    cfg.EVAL.USE_GPU = args.device != 'cpu'
    cfg.GPU_NUM = int(args.device != 'cpu')
    if args.training_seed is not None:
        cfg.ENV.SEED = args.training_seed
    daily, weekly = make_priors(args.bundle, args.output)
    cfg.MODEL.PARAM.init_path_daily, cfg.MODEL.PARAM.init_path_weekly = daily, weekly
    for section in (cfg.TRAIN, cfg.VAL, cfg.TEST):
        section.DATA.NUM_WORKERS = 0
        if args.smoke:
            section.DATA.BATCH_SIZE = 2
    if args.smoke:
        cfg.TRAIN.NUM_EPOCHS = 1
        cfg.TRAIN.LR_SCHEDULER.PARAM.epochs = 1
    torch.set_num_threads(4)
    # Explicit device setup avoids exposing the occupied GPU in CPU checks.
    set_device_type('cpu' if args.device == 'cpu' else 'gpu')
    cfg = init_cfg(cfg, save=True)
    runner = cfg.RUNNER(cfg)
    runner.train(cfg)
    paths = list(args.output.rglob('test_results.npz'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one final test export, got {paths}')
    return paths[0]


def export_common(args, result, meta):
    """Single metric policy across all models, alongside author native metrics."""
    test = arrays(args.bundle, 'test')
    with np.load(result) as native:
        pred = native['prediction'].copy()
        target = native['target'].copy()
        actual_inputs = native['inputs'].copy()
    if pred.shape != test['y'].shape or not np.isfinite(pred).all():
        raise ValueError('Prediction count/shape/finite check failed')
    np.testing.assert_allclose(target, test['y'], atol=1e-3, rtol=1e-5)
    np.testing.assert_allclose(actual_inputs[..., :1], test['x'], atol=1e-3, rtol=1e-5)
    def metrics(prediction, truth):
        keep = truth != 0
        if not keep.any():
            raise ValueError('All targets are zero')
        error = prediction[keep].astype('float64') - truth[keep]
        return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.square(error).mean())),
                    MAPE_percent=float((np.abs(error) / np.abs(truth[keep])).mean() * 100))
    report = dict(model=args.model, dataset=args.dataset, smoke=args.smoke,
                  training_seed=args.training_seed, device=args.device,
                  corruption=meta['key'], source_manifest_sha256=__import__('hashlib').sha256((args.bundle / 'manifest.json').read_bytes()).hexdigest(),
                  metric_policy='same for all models; target!=0; original units; MAPE in percent',
                  overall=metrics(pred, test['y']),
                  horizons={str(i + 1): metrics(pred[:, i], test['y'][:, i]) for i in range(12)})
    np.savez(args.output / 'robust_test_results.npz', inputs=test['x'], clean_inputs=test['clean_x'],
             prediction=pred, target=test['y'], corruption_mask=test['mask'], temporal_features=test['te'],
             history_start_index=test['starts'], forecast_start_index=test['starts'] + 12)
    (args.output / 'robust_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', choices=['STDN', 'STWave', 'HyperD'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.training_seed is None:
        args.training_seed = 233 if args.model == 'STDN' and args.dataset in ('PEMS03', 'PEMS08') else 1
    args.bundle, args.output = args.bundle.resolve(), args.output.resolve()
    if args.graph:
        args.graph = args.graph.resolve()
    # CUDA visibility is set before importing torch/model packages.
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
        # Some host runtimes cache CUDA availability before this script starts.
        # Scope the override to this dedicated worker, never the baseline job.
        import torch
        torch.cuda.is_available = lambda: False
    else:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError('CUDA was requested but is unavailable; choose --device cpu explicitly')
    meta = verify(args.bundle)
    if meta['key']['dataset'] != args.dataset or meta['key']['smoke'] != args.smoke:
        raise ValueError('Bundle dataset/smoke setting mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.rglob('*.pt')) or (args.output / 'robust_metrics.json').exists():
        raise FileExistsError('Output already contains model artifacts; choose a fresh output directory')
    result = {'STDN': stdn, 'STWave': stwave, 'HyperD': hyperd}[args.model](args, meta)
    export_common(args, result, meta)


if __name__ == '__main__':
    main()
