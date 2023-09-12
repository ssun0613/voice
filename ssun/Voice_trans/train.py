import os
import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from config import Config
from model.voice_trans import voice_trans as network

def setup(opt):
    #-------------------------------------------- setup device --------------------------------------------

    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # -------------------------------------------- setup network --------------------------------------------

    net = network().to(device)

    # -------------------------------------------- setup dataload --------------------------------------------

    from dataload_dacon import get_loader
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

    if opt.scheduler_name == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5)
    elif opt.scheduler_name == 'cycliclr':
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

    for curr_epoch in range(config.opt.epochs):
        print("-------------------------- Epoch : {} --------------------------".format(curr_epoch+1))
        for batch_id, data in enumerate(dataload, 1):

            # data : melsp, mfcc, pitch, len_org, sp_id
            voice = data['melsp'].to(device)
            sp_id = data['sp_id'].to(device)

            mel_output = net.forward(voice, sp_id)
