import math
import time
import torch
import random
import argparse
import numpy as np
import configparser
import json
from pathlib import Path
from tqdm import tqdm

from lib import utils
from model.models import STWave
from lib.graph_utils import loadGraph
from lib.utils import log_string, loadData, _compute_loss, metric, disentangle


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        self.trainX, self.trainY, self.trainTE, \
        self.valX, self.valY, self.valTE, \
        self.testX, self.testY, self.testTE, \
        self.mean, self.std, data = loadData(
                                        self.traffic_file, self.input_len, self.output_len,
                                        self.train_ratio, self.test_ratio, log)
        self.localadj, self.spawave, self.temwave = loadGraph(self.adj_file, self.tem_adj_file, self.heads*self.dims, data, log)
        log_string(log, '------------ End -------------\n')

        if self.smoke_batches:
            for split in ('train', 'val', 'test'):
                for suffix in ('X', 'Y', 'TE'):
                    key = split + suffix
                    setattr(self, key, getattr(self, key)[:self.smoke_batches*self.batch_size].copy())

        self.best_epoch = 0
        self.start_epoch = 1
        self.min_loss = 10000000.0
        self.train_order = np.arange(len(self.trainX))
        self.resume_state = None
        checkpoint = Path(self.output_dir, 'last_checkpoint.pt')
        if self.resume and checkpoint.exists():
            # This is a locally generated full training checkpoint, including
            # NumPy/Python RNG objects, not an externally downloaded weight file.
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['signature'] != self.resume_signature():
                raise ValueError('Resume checkpoint configuration does not match this run')
            self.localadj, self.spawave, self.temwave = state['graph_features']
            self.train_order = state['train_order']
            for key in ('trainX', 'trainY', 'trainTE'):
                setattr(self, key, getattr(self, key)[self.train_order])
            self.start_epoch = state['epoch'] + 1
            self.best_epoch = state['best_epoch']
            self.min_loss = state['min_loss']
            self.resume_state = state

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()
        if self.resume_state is not None:
            state = self.resume_state
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['scheduler'])
            self.atomic_save(state['best_model'], Path(self.model_file))

    def resume_signature(self):
        keys = ('traffic_file', 'adj_file', 'tem_adj_file', 'input_len', 'output_len',
                'train_ratio', 'test_ratio', 'heads', 'dims', 'layers', 'samples',
                'wave', 'level', 'batch_size', 'seed', 'smoke_batches')
        return dict(parameters={key: getattr(self, key) for key in keys},
                    torch_version=torch.__version__, numpy_version=np.__version__)

    @staticmethod
    def atomic_save(value, destination):
        temporary = destination.with_name(destination.name + '.tmp')
        torch.save(value, temporary)
        temporary.replace(destination)

    def save_training_state(self, epoch):
        state = dict(epoch=epoch, best_epoch=self.best_epoch, min_loss=self.min_loss,
                     signature=self.resume_signature(), model=self.model.state_dict(),
                     optimizer=self.optimizer.state_dict(), scheduler=self.lr_scheduler.state_dict(),
                     best_model=torch.load(self.model_file, map_location='cpu', weights_only=True),
                     train_order=self.train_order, graph_features=(self.localadj, self.spawave, self.temwave),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        self.atomic_save(state, Path(self.output_dir, 'last_checkpoint.pt'))
    
    def build_model(self):
        self.model = STWave(self.heads, self.dims, self.layers, self.samples,
                            self.localadj, self.spawave, self.temwave,
                            self.input_len, self.output_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                        mode='min', factor=0.1, patience=20,    
                                        threshold=0.001, threshold_mode='rel',
                                        cooldown=0, min_lr=2e-6, eps=1e-08)

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    XL, XH = disentangle(self.valX[start_idx : end_idx], self.wave, self.level)
                    Y = self.valY[start_idx : end_idx]
                    TE = torch.from_numpy(self.valTE[start_idx : end_idx,:,0,:]).to(self.device)
                    XL, XH = torch.from_numpy((XL-self.mean)/self.std).float().to(self.device), torch.from_numpy((XH-self.mean)/self.std).float().to(self.device)

                    y_hat, _ = self.model(XL, XH, TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        # for i in range(pred.shape[1]):
        #     mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
        #     maes.append(mae)
        #     rmses.append(rmse)
        #     mapes.append(mape)
        #     log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = self.min_loss
        num_train = self.trainX.shape[0]
        if self.resume_state is not None:
            state = self.resume_state
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
            self.resume_state = None
            log_string(log, f'Resuming full training state at epoch {self.start_epoch}')

        for epoch in tqdm(range(self.start_epoch,self.max_epoch+1)):
            self.model.train()
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            permutation = np.random.permutation(num_train)
            self.train_order = self.train_order[permutation]
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainTE = self.trainTE[permutation]
            num_batch = math.ceil(num_train / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    # disentangle x -> xl, xh
                    XL, XH = disentangle(self.trainX[start_idx : end_idx], self.wave, self.level)
                    YL, _ = disentangle(self.trainY[start_idx : end_idx], self.wave, self.level)
                    Y = torch.from_numpy(self.trainY[start_idx : end_idx]).float().to(self.device)
                    TE = torch.from_numpy(self.trainTE[start_idx : end_idx,:,0,:]).to(self.device)
                    XL, XH, YL = torch.from_numpy((XL-self.mean)/self.std).float().to(self.device), torch.from_numpy((XH-self.mean)/self.std).float().to(self.device), torch.from_numpy(YL).float().to(self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat, y_hat_l = self.model(XL, XH, TE)

                    loss = _compute_loss(Y, y_hat*self.std+self.mean) + _compute_loss(YL, y_hat_l*self.std+self.mean)

                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')

                    loss.backward()
                    if self.smoke_batches and not all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                        raise RuntimeError('Non-finite gradients')
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    n += Y.shape[0]
                    batch_count += 1
                    pbar.update(1)
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step(mae[-1])
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                self.atomic_save(self.model.state_dict(), Path(self.model_file))
            self.min_loss = min_loss
            self.start_epoch = epoch + 1
            self.save_training_state(epoch)
            Path(self.output_dir, 'progress.json').write_text(json.dumps(dict(
                epoch=epoch, max_epoch=self.max_epoch, best_epoch=self.best_epoch,
                best_validation_MAE=float(min_loss), updated=time.time())))
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device, weights_only=True))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    XL, XH = disentangle(self.testX[start_idx : end_idx], self.wave, self.level)
                    Y = self.testY[start_idx : end_idx]
                    TE = torch.from_numpy(self.testTE[start_idx : end_idx,:,0,:]).to(self.device)
                    XL, XH = torch.from_numpy((XL-self.mean)/self.std).float().to(self.device), torch.from_numpy((XH-self.mean)/self.std).float().to(self.device)

                    y_hat, _ = self.model(XL, XH, TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        # for i in range(pred.shape[1]):
        #     mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
        #     maes.append(mae)
        #     rmses.append(rmse)
        #     mapes.append(mape)
        #     log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))

        if not np.isfinite(pred).all():
            raise RuntimeError('Non-finite test prediction')
        horizons = {}
        for step in range(pred.shape[1]):
            h_mae, h_rmse, h_mape = metric(pred[:,step], label[:,step])
            horizons[str(step+1)] = dict(MAE=float(h_mae), RMSE=float(h_rmse), MAPE=float(h_mape))
        # The released loader leaves testX in original flow units. Keep windows
        # in chronological order and record source row indices for traceability.
        if self.history_start_file:
            history_start = np.load(self.history_start_file)[:len(pred)]
        else:
            total_steps = np.load(self.traffic_file)['data'].shape[0]
            test_start = total_steps - round(self.test_ratio * total_steps)
            history_start = test_start + np.arange(len(pred), dtype=np.int64)
        np.savez(Path(self.output_dir, 'test_results.npz'), inputs=self.testX,
                 prediction=pred, target=label, temporal_features=self.testTE[:,:,0,:],
                 history_start_index=history_start, forecast_start_index=history_start+self.input_len,
                 mean=self.mean, std=self.std)
        results = dict(seed=self.seed, smoke=bool(self.smoke_batches), best_epoch=self.best_epoch,
                       samples=len(pred), shape=list(pred.shape), batch_size=self.batch_size,
                       overall=dict(MAE=float(mae), RMSE=float(rmse), MAPE=float(mape)),
                       horizons=horizons, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if self.device.type=='cuda' else 0)
        Path(self.output_dir, 'test_metrics.json').write_text(json.dumps(results, indent=2))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
        return mae, rmse, mape

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args, _ = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--dims', type = int, default = config['param']['dims'])
    parser.add_argument('--heads', type = int, default = config['param']['heads'])
    parser.add_argument('--layers', type = int, default = config['param']['layers'])
    parser.add_argument('--wave', default = config['param']['wave'])
    parser.add_argument('--level', type = int, default = config['param']['level'])
    parser.add_argument('--samples', type = float, default = config['param']['samples'])

    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--tem_adj_file', default = config['file']['temadj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--history_start_file', default=None)
    parser.add_argument('--smoke_batches', type=int, default=0)
    parser.add_argument('--smoke_epochs', type=int, default=1)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    if args.smoke_batches < 0:
        parser.error('--smoke_batches must be nonnegative')
    if args.smoke_batches:
        args.max_epoch = args.smoke_epochs
    if args.output_dir is None:
        args.output_dir = str(Path(args.model_file).parent)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))

    solver.train()
    solver.test()
    
