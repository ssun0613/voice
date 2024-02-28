import os, sys, yaml
sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.FPF import FastPitchFormant
from model.speechsplit_ec import speechsplit, InterpLnr

from torch.utils.tensorboard import SummaryWriter
from functions.load_network import load_networks, init_weights
from functions.draw_function import tensorboard_draw
from functions.etc_fcn import quantize_f0_torch

import matplotlib
matplotlib.use("Agg")

preprocess_config = yaml.load(open("./config_load/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("./config_load/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader)

def setup(opt):
    # -------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        if not opt.debugging:
            device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # -------------------------------------------- setup dataload --------------------------------------------
    # remove noise dataset
    if not opt.debugging:
        from ssun.voice.data.dataload_remove_noise import get_loader
    else:
        from voice.data.dataload_remove_noise import get_loader
    dataload = get_loader(opt)

    # -------------------------------------------- setup network --------------------------------------------
    s_ = 'speechsplit_32'
    # pre_encoder = speechsplit().to(device)
    pre_encoder = load_networks(speechsplit().to(device), s_, device,  net_name='Generator', weight_path= "/storage/mskim/")
    FPF = FastPitchFormant(preprocess_config, model_config).to(device)

    if opt.continue_train:
        FPF = load_networks(FPF, opt.checkpoint_name, device, net_name='FPF', weight_path= "/storage/mskim/checkpoint/")
    # -------------------------------------------- setup optimizer --------------------------------------------
    if opt.optimizer_name == 'Adam':
        optimizer_FPF = torch.optim.Adam(FPF.parameters(), lr=opt.lr)
    else:
        optimizer_FPF = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))
    # -------------------------------------------- setup scheduler --------------------------------------------
    if opt.scheduler_name == 'cosine':
        scheduler_FPF = lr_scheduler.CosineAnnealingLR(optimizer_FPF, T_max=10, eta_min=1e-9)
    else:
        scheduler_FPF = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))

    return device, dataload, pre_encoder, FPF, optimizer_FPF, scheduler_FPF


if __name__ == "__main__":
    config = Config()
    config.print_options()

    setproctitle(config.opt.network_name)
    torch.cuda.set_device(int(config.opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter('/storage/mskim/tensorboard/{}'.format(config.opt.tensor_name))
    device, dataload, pre_encoder, FPF, optimizer_FPF, scheduler_FPF = setup(config.opt)

    mse_loss_sum = nn.MSELoss(reduction='sum')

    print(device)

    try:
        global_step = 0
        for curr_epoch in range(config.opt.epochs):
            print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch+1))
            for batch_id, data in enumerate(dataload, 1):
                global_step+=1

                mel_in = data['melsp'].to(device)
                mel_len = data['mel_len'].to(device)
                pitch_t = data['pitch'].to(device)
                len_org = data['len_org'].to(device)
                sp_id = data['sp_id'].to(device)
                sp_id_1 = data['sp_id_1'].to(device)

                # -------------- pretrain : content --------------

                x_f0 = torch.cat((mel_in, pitch_t), dim=-1)
                x_f0_intrp = InterpLnr()(x_f0, len_org)
                f0_org_intrp = quantize_f0_torch(x_f0_intrp[:, :, -1])[0]
                x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)

                pre_c = pre_encoder.forward(x_f0_intrp_org, mel_in, sp_id)

                # -------------- train : FastPitch-Formant --------------

                (mel_outs, p_predictions, log_d_predictions, d_rounded, src_masks, mel_masks, src_lens, mel_lens) = FPF.forward(sp_id_1, pre_c, p_targets=None, max_src_len = None, mels=None, mel_lens=None, max_mel_len=None,  d_targets=None, p_control=1.0, d_control=1.0)

                mel_loss = 0

                for mel_out in mel_outs:
                    mel_loss += mse_loss_sum(mel_out, mel_in)
                mel_loss = (mel_loss / (80 * mel_len)).mean()
                # mel_loss = mel_loss * 10

                optimizer_FPF.zero_grad()
                mel_loss.backward()
                torch.nn.utils.clip_grad_norm_(FPF.parameters(), max_norm=5)
                optimizer_FPF.step()

            if curr_epoch % 500 == 0:
                tensorboard_draw(writer, mel_in, mel_outs, mel_loss, global_step)

            writer.close()
            scheduler_FPF.step()

            print("mel_loss : %.5f\n" % mel_loss)

            if curr_epoch % 500 == 0:
                torch.save({'FPF': FPF.state_dict(), 'optimizer_FPF': optimizer_FPF.state_dict()}, "/storage/mskim/checkpoint/{}.pth".format(config.opt.checkpoint_name))

    except Exception as e:
        print(e)
