"""Convert external BasicTS or raw PeMS data into STDN sample files."""
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np

NODES = {'PEMS03':358, 'PEMS04':307, 'PEMS07':883, 'PEMS08':170}


def dataset_folder(root, dataset):
    root = Path(root).expanduser().resolve()
    for candidate in (root/dataset, root/f'PeMSD{int(dataset[-2:])}', root):
        if any((candidate/name).exists() for name in ('desc.json','samples_12_12_5.npz',f'{dataset}.npz',f'{dataset}.csv')):
            return candidate
    raise FileNotFoundError(f'{dataset} was not found under {root}')


def prepare(dataset, data_root, raw_root, destination, smoke=False, start_date=None):
    source = dataset_folder(data_root, dataset)
    raw = dataset_folder(raw_root or data_root, dataset)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    samples = destination/'samples_12_12_5.npz'
    graph = destination/f'{dataset}.csv'
    ready = source/'samples_12_12_5.npz'
    if ready.exists() and (raw/f'{dataset}.csv').exists():
        # Prepared samples require a graph already indexed in sample node order.
        with (raw/f'{dataset}.csv').open(newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            edges = [row for row in reader if len(row) == 3]
        if not edges or any(not (0 <= int(row[0]) < NODES[dataset] and
                                 0 <= int(row[1]) < NODES[dataset]) for row in edges):
            raise ValueError('Prepared samples require a graph with contiguous node indices in sample order; convert the raw graph first.')
        return ready, raw/f'{dataset}.csv'
    if (source/'desc.json').exists():
        desc = json.loads((source/'desc.json').read_text(encoding='utf-8-sig'))
        if desc['feature_description'] != ['traffic flow','time of day','day of week']:
            raise ValueError('Expected BasicTS features: traffic flow, time of day, day of week')
        data_file = source/'data.dat'
        if data_file.stat().st_size != int(np.prod(desc['shape']))*4:
            raise ValueError('data.dat size does not match desc.json')
        data = np.memmap(data_file, dtype='float32', mode='r', shape=tuple(desc['shape']))
        flow = data[...,:1]
        temporal = np.column_stack([np.rint(data[:,0,2]*7),np.rint(data[:,0,1]*288)]).astype('int32')
        inputs = [source/'desc.json', data_file]
    else:
        if start_date is None:
            raise ValueError('Raw NPZ needs --start-date YYYY-MM-DD to encode the actual weekday; use BasicTS data to avoid this requirement.')
        data_file = source/f'{dataset}.npz'
        with np.load(data_file) as archive:
            flow = archive['data'][...,:1].astype('float32')
        start = datetime.fromisoformat(start_date)
        if start.second or start.minute % 5:
            raise ValueError('--start-date must align to a 5-minute time slot')
        stamps = [start+timedelta(minutes=5*i) for i in range(len(flow))]
        temporal = np.array([(s.weekday(),s.hour*12+s.minute//5) for s in stamps],dtype='int32')
        inputs = [data_file]
    if flow.shape[1] != NODES[dataset]:
        raise ValueError(f'{dataset} needs {NODES[dataset]} nodes, got {flow.shape[1]}')
    if not np.isfinite(flow).all():
        raise ValueError('Non-finite flow data')
    if np.any((temporal[:,0]<0)|(temporal[:,0]>6)|(temporal[:,1]<0)|(temporal[:,1]>287)):
        raise ValueError('Invalid temporal features')
    csv_path = raw/f'{dataset}.csv'
    inputs.append(csv_path)
    mapping_file = raw/'PEMS03.txt'
    if dataset == 'PEMS03':
        inputs.append(mapping_file)
    key = dict(dataset=dataset, smoke=smoke, start_date=start_date,
               sources=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs])
    cache_info = destination/'preparation.json'
    if samples.exists() and graph.exists() and cache_info.exists():
        if json.loads(cache_info.read_text())['cache_key'] == key:
            return samples,graph
    n = len(flow)
    train,valid = int(n*.6),int(n*.2)
    arrays = {}
    for split,begin,end in [('train',0,train),('val',train,train+valid),('test',train+valid,n)]:
        count = end-begin-23
        if count < 1:
            raise ValueError(f'{split} has fewer than 24 time steps')
        indices = np.linspace(begin,end-24,min(count,4),dtype=int) if smoke else np.arange(begin,end-23)
        arrays[f'{split}_x'] = np.stack([flow[i:i+12] for i in indices])
        arrays[f'{split}_target'] = np.stack([flow[i+12:i+24] for i in indices])
        arrays[f'{split}TE'] = np.stack([temporal[i:i+24] for i in indices])
    print(f'Preparing {dataset}: {len(arrays["train_x"])} training windows',flush=True)
    np.savez_compressed(samples, **arrays)
    mapping = None
    if dataset == 'PEMS03':
        mapping = {int(x):i for i,x in enumerate(mapping_file.read_text().split())}
    edges = []
    with csv_path.open(newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if len(row) != 3:
                continue
            i,j=int(row[0]),int(row[1])
            if mapping is not None:
                i,j=mapping[i],mapping[j]
            if not (0<=i<NODES[dataset] and 0<=j<NODES[dataset]):
                raise ValueError(f'Invalid edge: {row}')
            edges.append((i,j,float(row[2])))
    if not edges:
        raise ValueError('The graph CSV has no edges')
    with graph.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['from','to','distance'])
        writer.writerows(edges)
    cache_info.write_text(json.dumps(dict(cache_key=key,arrays={k:list(v.shape) for k,v in arrays.items()}),indent=2))
    return samples,graph
