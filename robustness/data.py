"""Shared windows and persisted perturbations, always in original flow units."""
import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test')
NODES = {'PEMS03': 358, 'PEMS04': 307, 'PEMS07': 883, 'PEMS08': 170}
VERSION = 1


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def corrupt(x, kind, rate, seed=2024):
    """Return a copy, selection mask, and selected nodes. Matches supplied code.

    Mask counts selected positions, including positions already zero. The same
    seed is intentionally restarted in each split, as in the supplied util.py.
    Noise/gap locations differ with split shape; node IDs remain identical.
    """
    if x.ndim != 4 or x.shape[-1] < 1 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError('Expected floating [sample,time,node,feature] data')
    if not 0 <= rate <= 1 or kind not in ('clean', 'node', 'node_gap', 'noise'):
        raise ValueError('Invalid corruption or rate')
    out = x.copy()
    s, t, n, _ = x.shape
    mask = np.zeros((s, t, n), dtype=bool)
    rng = np.random.RandomState(seed)
    nodes = np.empty(0, dtype=np.int64)
    if kind == 'node':
        nodes = rng.permutation(n)[:int(n * rate)]
        mask[:, :, nodes] = True
    elif kind == 'node_gap':
        if t % 3:
            raise ValueError('History length must be divisible by block_length=3')
        blocks = t // 3
        for node in range(n):
            selected = rng.choice(s * blocks, int(s * blocks * rate), replace=False)
            for offset in range(3):
                mask[selected // blocks, (selected % blocks) * 3 + offset, node] = True
    elif kind == 'noise':
        selected = rng.choice(mask.size, int(mask.size * rate), replace=False)
        mask.reshape(-1)[selected] = True
        flow = out[..., 0].copy().reshape(-1)
        flow[selected] += rng.normal(10, 500, len(selected))
        out[..., 0] = np.maximum(flow.reshape(mask.shape), 0)
    if kind in ('node', 'node_gap'):
        out[..., 0][mask] = 0
    return out, mask, nodes


def prepare(data_root, dataset, destination, kind, rate, seed=2024, smoke=False):
    """Build once, share between all three models; no changes to source files."""
    if dataset not in NODES:
        raise ValueError(dataset)
    source = Path(data_root).resolve() / dataset
    desc_path, data_path = source / 'desc.json', source / 'data.dat'
    desc = json.loads(desc_path.read_text(encoding='utf-8-sig'))
    if desc['feature_description'] != ['traffic flow', 'time of day', 'day of week']:
        raise ValueError('Expected BasicTS flow/TOD/DOW channels')
    shape = tuple(desc['shape'])
    if shape[1:] != (NODES[dataset], 3) or data_path.stat().st_size != np.prod(shape) * 4:
        raise ValueError('Dataset shape or file length mismatch')
    stat = data_path.stat()
    key = dict(version=VERSION, dataset=dataset, kind=kind, rate=rate, corruption_seed=seed,
               smoke=smoke, data_path=str(data_path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
               desc_sha256=digest(desc_path))
    destination = Path(destination).resolve()
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['key'] != key:
            raise ValueError(f'Existing bundle has different settings: {destination}')
        if any(not (destination / name).exists() for name in old['sha256']):
            raise ValueError(f'Incomplete bundle: {destination}')
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    raw = np.memmap(data_path, mode='r', dtype='float32', shape=shape)
    a, b = int(shape[0] * .6), int(shape[0] * .2)
    bounds = [('train', 0, a), ('val', a, a + b), ('test', a + b, shape[0])]
    metadata = dict(key=key, splits={}, sha256={}, protocol='input-only; 6:2:2 then windows; 12->12',
                    noise_mean=10, noise_std=500, noise_clip_min=0, block_length=3,
                    perturbation_unit='sample,history-step,node; overlapping windows may differ',
                    node_rounding='floor(N*rate)', per_node_gap_rounding='floor(S*4*rate)*3/(S*12)')
    # Reconstruct training observations for data-dependent graph/prior building.
    # Only corrupted historical inputs contribute; future labels never contribute.
    observed_sum = np.zeros((a, shape[1]), dtype=np.float64)
    observed_count = np.zeros(a, dtype=np.int64)
    for split, begin, end in bounds:
        if end - begin < 24:
            raise ValueError(f'{split} needs at least 24 time steps')
        starts = (np.linspace(begin, end - 24, 4, dtype=np.int64) if smoke
                  else np.arange(begin, end - 23, dtype=np.int64))
        x = np.stack([raw[i:i + 12, :, :1] for i in starts])
        y = np.lib.format.open_memmap(destination / f'{split}_y.npy', mode='w+',
                                    dtype='float32', shape=x.shape)
        te = np.empty((len(starts), 24, 2), dtype=np.int32)
        for row, index in enumerate(starts):
            y[row] = raw[index + 12:index + 24, :, :1]
            te[row, :, 0] = np.rint(raw[index:index + 24, 0, 2] * 7)
            te[row, :, 1] = np.rint(raw[index:index + 24, 0, 1] * 288)
        y.flush()
        del y
        perturbed, mask, nodes = corrupt(x, kind, rate, seed)
        for name, array in [('x', perturbed), ('clean_x', x), ('mask', mask), ('te', te), ('starts', starts)]:
            np.save(destination / f'{split}_{name}.npy', array)
        metadata['splits'][split] = dict(samples=len(starts), begin=begin, end=end,
                     selected_count=int(mask.sum()), selected_fraction=float(mask.mean()),
                     per_node_selected_count=mask.sum(axis=(0, 1)).tolist(), missing_nodes=nodes.tolist())
        if split == 'train':
            mean = float(np.mean(perturbed, dtype=np.float64))
            std = float(np.std(perturbed, dtype=np.float64))
            metadata['scaler'] = dict(mean=mean, std=std if std > 0 else 1.,
                                      fitted_on='corrupted training input windows only', ddof=0)
            for row, index in enumerate(starts):
                observed_sum[index:index + 12] += perturbed[row, ..., 0]
                observed_count[index:index + 12] += 1
        del perturbed, mask, x
    valid = observed_count > 0
    observed = observed_sum[valid] / observed_count[valid, None]
    np.save(destination / 'train_observed.npy', observed.astype('float32'))
    np.save(destination / 'train_observed_indices.npy', np.flatnonzero(valid))
    for path in sorted(destination.glob('*.npy')):
        metadata['sha256'][path.name] = digest(path)
    text = json.dumps(metadata, indent=2)
    temporary = destination / 'manifest.tmp'
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(manifest_path)
    return destination


def arrays(bundle, split):
    bundle = Path(bundle)
    return {key: np.load(bundle / f'{split}_{key}.npy', mmap_mode='r')
            for key in ('x', 'y', 'te', 'starts', 'mask', 'clean_x')}


def verify(bundle):
    bundle = Path(bundle)
    meta = json.loads((bundle / 'manifest.json').read_text())
    for name, expected in meta['sha256'].items():
        if digest(bundle / name) != expected:
            raise ValueError(f'Bundle hash mismatch: {name}')
    return meta
