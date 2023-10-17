import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.voice_trans import voice_trans as network

import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def setup(opt):
    #-------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # -------------------------------------------- setup network --------------------------------------------
    net = network(opt, device).to(device)
    # -------------------------------------------- setup dataload --------------------------------------------
    if not opt.debugging:
        from ssun.Voice_trans.data.dataload_dacon import get_loader
    else:
        from Voice_trans.data.dataload_dacon import get_loader
    dataload = get_loader(opt)
    # -------------------------------------------- setup optimizer --------------------------------------------
    if opt.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    elif opt.optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr)
    else:
        optimizer = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))
    # -------------------------------------------- setup scheduler --------------------------------------------
    if opt.scheduler_name == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-9, max_lr=opt.lr, cycle_momentum=False, step_size_up=3, step_size_down=17, mode='triangular2')
    elif opt.scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-9)
    else:
        scheduler = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))

    return device, net, dataload, optimizer, scheduler

if __name__ == "__main__":
    config = Config()
    device, net, dataload, optimizer, scheduler = setup(config.opt)
    setproctitle(config.opt.network_name)

    loss_m = nn.MSELoss(reduction='sum')
    loss_l = nn.L1Loss(reduction='sum')
    print("loss_m = nn.MSELoss(reduction='sum')")
    print("loss_l = nn.L1Loss(reduction='sum')")

    for curr_epoch in range(config.opt.epochs):
        step = 0
        print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch+1))
        for batch_id, data in enumerate(dataload, 1):
            step +=1

            # data : melsp, mfcc, pitch, len_org, sp_id
            voice = data['melsp'].to(device)
            pitch_t = data['pitch'].to(device)
            sp_id = data['sp_id'].to(device)

            mel_output, pitch_p, rhythm, content, rhythm_l, content_l = net.forward(voice, sp_id)

            voice_loss = loss_m(voice, mel_output)
            rhythm_loss = loss_l(rhythm, rhythm_l)
            content_loss = loss_l(content, content_l)

            recon_loss = voice_loss + (config.opt.lambda_r * rhythm_loss) + (config.opt.lambda_c * content_loss)
            pitch_loss = loss_m(pitch_t, pitch_p)
            total_loss = recon_loss + pitch_loss

            if (torch.isnan(voice_loss).int().sum().item()!=0) or (torch.isnan(pitch_loss).int().sum().item()!=0):
                print("epoch : {}, step : {}".format(curr_epoch+1, step))
                print("mel_output : {}".format(torch.isnan(mel_output).int().sum().item()))
                print("voice_loss : {}".format(torch.isnan(voice_loss).int().sum().item()))
                print("pitch_p : {}".format(torch.isnan(pitch_p).int().sum().item()))
                print("pitch_t isnan : {}".format(torch.isnan(pitch_t).int().sum().item()))
                print("pitch_t isinf : {}".format(torch.isinf(pitch_t).int().sum().item()))
                print("pitch_loss : {}".format(torch.isnan(pitch_loss).int().sum().item()))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.autograd.set_detect_anomaly(True)


        scheduler.step()
        print("voice_loss : {:.5f} rhythm_loss : {:.5f} content_loss : {:.5f} recon_loss : {:.5f}\n".format(voice_loss, rhythm_loss, content_loss, recon_loss))
        print("pitch_loss : %.5lf\n" % pitch_loss)
        print("total_loss : %.5lf\n" % total_loss)
        print("Learning rate : %.5f\n" % optimizer.param_groups[0]['lr'])
