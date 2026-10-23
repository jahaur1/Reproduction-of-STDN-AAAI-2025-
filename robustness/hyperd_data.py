"""BasicTS adapters for the shared robustness bundle."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import arrays


class BundleDataset(Dataset):
    def __init__(self, bundle, mode, **kwargs):
        self.values = arrays(bundle, 'val' if mode == 'valid' else mode)

    def __len__(self):
        return len(self.values['x'])

    def __getitem__(self, index):
        item = self.values
        n = item['x'].shape[2]
        x = np.empty((12, n, 3), dtype='float32')
        y = np.empty_like(x)
        x[..., :1], y[..., :1] = item['x'][index], item['y'][index]
        te = item['te'][index]
        x[..., 1] = te[:12, 1, None] / 288
        x[..., 2] = te[:12, 0, None] / 7
        y[..., 1] = te[12:, 1, None] / 288
        y[..., 2] = te[12:, 0, None] / 7
        return dict(inputs=x, target=y)


class BundleScaler:
    def __init__(self, bundle):
        self.rescale = True
        self.stats = json.loads((Path(bundle) / 'manifest.json').read_text())['scaler']

    def transform(self, value):
        value = value.clone()
        value[..., 0] = (value[..., 0] - self.stats['mean']) / self.stats['std']
        return value

    def inverse_transform(self, value):
        value = value.clone()
        value[..., 0] = value[..., 0] * self.stats['std'] + self.stats['mean']
        return value


def make_priors(bundle, out):
    """Fit periodic priors only to corrupted historical training observations."""
    bundle, out = Path(bundle), Path(out)
    meta = json.loads((bundle / 'manifest.json').read_text())
    source = Path(meta['key']['data_path'])
    desc = json.loads(source.with_name('desc.json').read_text())
    raw = np.memmap(source, mode='r', dtype='float32', shape=tuple(desc['shape']))
    indices = np.load(bundle / 'train_observed_indices.npy')
    flow = np.load(bundle / 'train_observed.npy')
    flow = (flow - meta['scaler']['mean']) / meta['scaler']['std']
    daily = np.rint(raw[indices, 0, 1] * 288).astype(int)
    weekly = daily + np.rint(raw[indices, 0, 2] * 7).astype(int) * 288
    result = []
    for name, bins, count in [('daily_init.npy', daily, 288), ('weekly_init.npy', weekly, 2016)]:
        sums = np.zeros((count, flow.shape[1]), dtype='float64')
        counts = np.bincount(bins, minlength=count)
        np.add.at(sums, bins, flow)
        # Unobserved time bins use normalized zero, never test/validation values.
        sums /= np.maximum(counts, 1)[:, None]
        path = out / name
        np.save(path, sums.astype('float32'))
        result.append(str(path))
    return result
