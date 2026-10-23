"""One-command data preparation and STDN training (no datasets bundled)."""
import argparse
import configparser
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from training_options import resolve_seed

ROOT = Path(__file__).resolve().parent
DATASETS = ('PEMS03','PEMS04','PEMS07','PEMS08')


def dependencies(install):
    modules=('torch','numpy','pandas','scipy','sklearn','torch_geometric')
    missing=[m for m in modules if importlib.util.find_spec(m) is None]
    if missing and install:
        subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements.txt')],check=True)
        missing=[m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f'Missing dependencies: {missing}. Use --install-deps or install requirements.txt into this Python environment.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS+('all',),default='PEMS04')
    parser.add_argument('--data-root',type=Path,required=True,help='External BasicTS, raw NPZ, or prepared STDN dataset directory')
    parser.add_argument('--raw-root',type=Path,help='External graph CSV and PEMS03.txt directory; defaults to data-root')
    parser.add_argument('--start-date',help='Required for raw NPZ only; actual first time point in ISO format')
    parser.add_argument('--output-root',type=Path,default=ROOT/'results')
    parser.add_argument('--cache-root',type=Path,default=ROOT/'.cache')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--seed',type=int,default=None,help='Override config seed; defaults: 03/08=233, 04/07=unset')
    parser.add_argument('--batch-size',type=int,default=None,help='Omit to preserve config batch=64')
    parser.add_argument('--smoke',action='store_true',help='One epoch, batch=2, at most 4 windows per split; not a full reproduction')
    parser.add_argument('--install-deps',action='store_true',help='Install missing dependencies into this Python environment')
    args=parser.parse_args()
    dependencies(args.install_deps)
    from prepare_data import prepare
    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install a CUDA-enabled PyTorch build or choose --device cpu.')
    args.data_root=args.data_root.expanduser().resolve()
    if args.raw_root:
        args.raw_root=args.raw_root.expanduser().resolve()
    args.output_root=args.output_root.expanduser().resolve()
    args.cache_root=args.cache_root.expanduser().resolve()
    datasets=DATASETS if args.dataset=='all' else (args.dataset,)
    mode='smoke' if args.smoke else 'full'
    output=args.output_root/mode
    output.mkdir(parents=True,exist_ok=True)
    # Never let two launchers overwrite the same results on either platform.
    lock=(output/'.run.lock').open('a+b')
    if lock.tell()==0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        raise RuntimeError(f'Another run already owns {output}') from None
    summary=[]
    for dataset in datasets:
        destination=output/dataset
        destination.mkdir(exist_ok=True)
        metrics=destination/'metrics.json'
        metrics.unlink(missing_ok=True)
        status=dict(dataset=dataset,mode=mode,status='preparing',started=time.time())
        state_path=destination/'status.json'
        def state(**values):
            status.update(values)
            state_path.write_text(json.dumps(status,indent=2))
        try:
            samples,graph=prepare(dataset,args.data_root,args.raw_root,args.cache_root/mode/dataset,args.smoke,args.start_date)
            cfg=configparser.ConfigParser()
            cfg.read(ROOT/'conf'/f'PEMSD{int(dataset[-2:])}_1dim_12.conf')
            cfg['Data']['graph_signal_matrix_filename']=str(samples)
            cfg['Data']['distance_csv_path']=str(graph)
            cfg['Training']['gpu']=str(int(device=='cuda'))
            cfg['Training']['ctx']='0'
            effective_seed = resolve_seed(cfg['Training'], args.seed)
            if effective_seed is not None:
                cfg['Training']['seed'] = str(effective_seed)
            runtime=destination/'runtime.conf'
            with runtime.open('w') as stream:
                cfg.write(stream)
            environment=dict(python=sys.version,torch=torch.__version__,device=device,seed=effective_seed,
                             smoke=args.smoke,config_provenance='Torch-MTS parameter reference (2db4de371584067160f9a37f1ae59495699b4a0a); local author data/trainer/metrics retained' if dataset in ('PEMS03','PEMS08') else 'author config with local device/path adaptation')
            (destination/'environment.json').write_text(json.dumps(environment,indent=2))
            command=[sys.executable,'-u',str(ROOT/'train.py'),'--config',str(runtime),'--device',device,'--output-dir',str(destination)]
            if args.smoke:
                command+=['--smoke-batches','2','--batch-size','2']
            elif args.batch_size:
                command+=['--batch-size',str(args.batch_size)]
            if effective_seed is not None:
                command+=['--seed',str(effective_seed)]
            state(status='training',device=device)
            print(f'{dataset}: training on {device}; log: {destination/"console.log"}',flush=True)
            with (destination/'console.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(metrics.read_text())
            if not report['checkpoint_reload_passed']:
                raise RuntimeError('Checkpoint reload check failed')
            state(status='complete',finished=time.time(),overall=report['overall'])
            summary.append(report)
            (output/'summary.json').write_text(json.dumps(summary,indent=2))
            print(f'{dataset}: complete; metrics: {metrics}',flush=True)
        except Exception as exc:
            state(status='failed',finished=time.time(),error=str(exc))
            raise


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,FileNotFoundError,ValueError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
