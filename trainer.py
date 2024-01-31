import os, sys, time
from pesq import pesq, cypesq
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.GFT_conformer import *
from dataLoader import *

from utils.early_stopping import *
from utils.compute_metrics import *
from utils.pmsqe_loss import *
from utils.losses import *
from utils.utils import *
from utils.gsp import *

import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        self.init_dataloader()
        self.model = GFT_conformer(
            kernel_num=tuple_data(self.config['model']['kernel_num']),
            device=self.device, n_f=self.config['n_f'], win_l=self.config['win_l'], n_s=self.config['n_s'],
            U_path=self.config['model_path'], training=self.config['load_model']
        ).to(self.device)
        self.loss_fn = si_snr_loss
        self.loss_fn_pesq = pmsqe_loss_fn(self.config)
        self.loss_snr = 0.10
        self.loss_pesq = 0.95

        optimizer = getattr(sys.modules['torch.optim'], self.config['optimizer'])
        self.optimizer = optimizer(self.model.parameters(), lr=self.config['learning_rate'])

        # 训练策略
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer, factor=self.config['scheduler']['factor'],
            patience=self.config['scheduler']['patience'], verbose=self.config['scheduler']['verbose']
        )

        self.early_stop = EarlyStopping(verbose=True, delta=1e-7)

    def init_dataloader(self):
        if self.config['VCTK_dataset']['used']:
            dataset_config = self.config['VCTK_dataset']
            train_dataset = VCTKTrain(dataset_config['path'], dataset_config['trainset'],
                                      wav_dur=dataset_config['wav_dur'], is_trian=True)
            self.train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'],
                                           num_workers=self.config['train']['num_workers'], shuffle=True)

            validate_dataset = VCTKTrain(dataset_config['path'], dataset_config['trainset'],
                                      wav_dur=dataset_config['wav_dur'], is_trian=False)
            self.validate_loader = DataLoader(validate_dataset, batch_size=self.config['train']['batch_size'],
                                              num_workers=self.config['train']['num_workers'])

            self.eval_loaders = []
            eval_dataset = VCTKEval(dataset_config['path'], dataset_config['testset'])
            eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=self.config['train']['num_workers'])
            self.eval_loaders.append(eval_loader)

        elif self.config['DeepXi_dataset']['used']:
            dataset_config = self.config['DeepXi_dataset']
            train_dataset = DeepXiTrain(dataset_config['path'], dataset_config['train_clean_files'],
                                               dataset_config['train_noise_files'], wav_dur=dataset_config['wav_dur'])
            self.train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'],
                                           num_workers=self.config['train']['num_workers'], shuffle=True)

            validate_dataset = DeepXiVal(dataset_config['path'], dataset_config['validate_files'],
                                                wav_dur=dataset_config['wav_dur'])
            self.validate_loader = DataLoader(validate_dataset, batch_size=self.config['train']['batch_size'],
                                              num_workers=self.config['train']['num_workers'])

            snrs = tuple_data(dataset_config['test_snrs'])
            self.eval_loaders = []
            for snr in snrs:
                eval_dataset = DeepXiEval(dataset_config['path'], dataset_config['test_files'], snr)
                eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=self.config['train']['num_workers'])
                self.eval_loaders.append(eval_loader)
        
    def train_epoch(self, loss_snr=0.1, loss_pesq=0.9):
        self.model.train()
        train_ep_loss = 0.
        counter = 0
        for noisy_x, clean_x in self.train_loader:
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)

            # Normalization
            if self.config['VCTK_dataset']['used']:
                c = torch.sqrt(noisy_x.size(-1) / torch.sum((noisy_x ** 2.0), dim=-1))
                noisy_x, clean_x = torch.transpose(noisy_x, 0, 1), torch.transpose(clean_x, 0, 1)
                noisy_x, clean_x = torch.transpose(noisy_x * c, 0, 1), torch.transpose(clean_x * c, 0, 1)

            # zero  gradients
            self.model.zero_grad()

            # get the output from the model
            pred_x, pred_graph = self.model(noisy_x)

            # calculate loss
            snr_loss = self.loss_fn(pred_x, clean_x)
            pesq_loss = self.loss_fn_pesq(pred_x, clean_x)
            loss = loss_snr * snr_loss + loss_pesq * pesq_loss
            loss.backward()
            self.optimizer.step()

            train_ep_loss += loss.item()
            counter += 1

        clear_cache()
        return train_ep_loss / counter

    def test_epoch(self):
        self.model.eval()
        val_ep_loss_snr = 0.
        val_ep_loss_pesq = 0.
        pesq_total = 0.
        counter = 0.
        for noisy_x, clean_x in self.validate_loader:
            noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)

            # Normalization
            if self.config['VCTK_dataset']['used']:
                c = torch.sqrt(noisy_x.size(-1) / torch.sum((noisy_x ** 2.0), dim=-1))
                noisy_x, clean_x = torch.transpose(noisy_x, 0, 1), torch.transpose(clean_x, 0, 1)
                noisy_x, clean_x = torch.transpose(noisy_x * c, 0, 1), torch.transpose(clean_x * c, 0, 1)

            # get the output from the model
            pred_x, pred_graph = self.model(noisy_x)

            # calculate loss
            snr_loss = self.loss_fn(pred_x, clean_x)
            pesq_loss = self.loss_fn_pesq(pred_x, clean_x)
            val_ep_loss_snr += snr_loss.item()
            val_ep_loss_pesq += pesq_loss.item()
            counter += 1

            clean_x = clean_x.detach().cpu().numpy()
            pred_x = pred_x.detach().cpu().numpy()

            # 计算pesq
            try:
              psq = 0.
              for i in range(len(clean_x)):
                  psq += pesq(16000, clean_x[i], pred_x[i], 'wb')  # WB-PESQ 宽带
              psq /= len(clean_x)
              pesq_total += psq
            except cypesq.NoUtterancesError:
              pass

        clear_cache()
        return val_ep_loss_snr / counter, val_ep_loss_pesq / counter, pesq_total / counter

    def train(self):
        for e in range(self.config['train']['epochs']):
            start = time.time()

            train_loss = self.train_epoch(loss_snr=self.loss_snr, loss_pesq=self.loss_pesq)
            with torch.no_grad():
                test_loss_snr, test_loss_pesq, test_pesq = self.test_epoch()

            self.scheduler.step(self.loss_snr * test_loss_snr + self.loss_pesq * test_loss_pesq)

            end = time.time()

            print("Epoch: {}/{}...".format(e + 1, self.config['train']['epochs']),
                  "Loss: {:.6f}...".format(train_loss),
                  "Test snr_Loss: {:.6f}...".format(test_loss_snr),
                  "Test pesq_Loss: {:.6f}...".format(test_loss_pesq),
                  "Test Pesq: {:.6f}...".format(test_pesq),
                  "time: {:.1f}min".format((end - start) / 60))

            # self.save(self.config['model_name'] + '_state_last.pth')
            self.save(self.config['model_name'] + f'_state{e + 1}.pth')
            
            self.early_stop(self.loss_snr * test_loss_snr + self.loss_pesq * test_loss_pesq, self.model, self.config['model_path'], name=self.config['model_name'])
            
            if self.early_stop.early_stop:
                print("Early stopping!")
                break


    def eval(self):
        self.model.eval()

        print("\n\nModel evaluation.\n")

        metricss = []
        for index, loader in enumerate(self.eval_loaders):
            counter = 0.
            metrics_total = np.zeros(6) 

            start = time.time()
            for noisy_x, clean_x, filename in loader:
                noisy_x, clean_x = noisy_x.to(self.device), clean_x.to(self.device)

                if self.config['VCTK_dataset']['used']:
                    c = torch.sqrt(noisy_x.size(-1) / torch.sum((noisy_x ** 2.0), dim=-1))
                    noisy_x = torch.transpose(noisy_x, 0, 1)
                    noisy_x = torch.transpose(noisy_x * c, 0, 1)
                
                with torch.no_grad():
                    pred_x, pred_graph = self.model(noisy_x)
                    clean_x = clean_x[:, :pred_x.shape[1]]

                clean_x = clean_x.detach().cpu().numpy()
                noisy_x = noisy_x.detach().cpu().numpy()
                pred_x = pred_x.detach().cpu().numpy()
                
                # 计算指标
                metrics = compute_metrics(clean_x[0], pred_x[0], 16000, 0)
                metrics = np.array(metrics)

                metrics_total = metrics_total + metrics
                counter += 1

            metricss.append(metrics_total / counter)
            end = time.time()

            print("Dataset[{}]...".format(index),
                  "PESQ: {:.6f}...".format(metricss[index][0]),
                  "CSIG: {:.6f}...".format(metricss[index][1]),
                  "CBAK: {:.6f}...".format(metricss[index][2]),
                  "COVL: {:.6f}...".format(metricss[index][3]),
                  "STOI: {:.6f}...".format(metricss[index][5]),
                  "time: {:.1f}min".format((end - start) / 60))
        
        metrics_avg = np.zeros(6) 
        for temp in metricss:
            metrics_avg += temp
        metrics_avg = metrics_avg / len(self.eval_loaders)
          
        print("Average...",
              "PESQ: {:.6f}...".format(metrics_avg[0]),
              "CSIG: {:.6f}...".format(metrics_avg[1]),
              "CBAK: {:.6f}...".format(metrics_avg[2]),
              "COVL: {:.6f}...".format(metrics_avg[3]),
              "STOI: {:.6f}...".format(metrics_avg[5]))

    def save(self, pth_name='model_state.pth'):
        os.makedirs(self.config['model_path'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['model_path'], pth_name))

    def load(self, model_state):
        self.model.load_state_dict(torch.load(os.path.join(self.config['model_path'], model_state)))