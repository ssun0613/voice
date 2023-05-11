from models_pro import Generator_3 as Generator
from models_pro import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy, plot_spectrogram


class Solver(object):
    def __init__(self, vcc_loader, config, hparams):
        # Data loader
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Miscellaneous
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_Dir = config.model_save_dir

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        self.G = Generator(self.hparams)

        self.Interp = InterpLnr(self.hparams)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])

        self.G.to(self.device)
        self.Interp.to(self.device)

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def train(self):
        data_loader = self.vcc_loader
        data_iter = iter(data_loader)

        start_iters = 0
        if self.resume_iters:
            print('Resuming...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)

        g_lr = self.g_lr
        print('Current learning rate, g_lr: {}.'.format(g_lr))

        keys = ['G/loss_id']

        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # ------- 1. Preprocess input data -------
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)

            x_real_org = x_real_org.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)

            # ------- 2. train the generator -------
            self.G = self.G.train()

            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            x_f0_intrp = self.Interp(x_f0, len_org)
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:, :, -1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)

            x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean')

            g_loss = g_loss_id
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            loss = {}
            loss['G/loss_id'] = g_loss_id.item()

            # ------- Miscellaneous -------
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i+1)
                    self.writer.add_figure('generated/mel_spectorgram_num_iters_{}'.format(i),plot_spectrogram(x_identic[0].squeeze(0).cpu().numpy()), i)
                    self.writer.add_figure('gt/mel_spectorgram_num_iters_{}'.format(i),plot_spectrogram(x_real_org[0].squeeze(0).cpu().numpy()), i)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                torch.save({'model': self.G.state_dict(),'optimizer': self.g_optimizer.state_dict()}, G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

