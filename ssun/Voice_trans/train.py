import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.voice_trans import voice_trans as network

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf

import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def setup(opt):
    #-------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        if not opt.debugging:
            device = torch.device("cpu")
            # device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
        else:
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
    writer = SummaryWriter()
    config.print_options()
    torch.cuda.set_device(int(config.opt.gpu_id))
    device, net, dataload, optimizer, scheduler = setup(config.opt)
    setproctitle(config.opt.network_name)

    loss_m = nn.MSELoss(reduction='mean')
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

            # mel_output, pitch_p, rhythm, content, rhythm_r, content_r = net.forward(voice, sp_id)
            mel_output, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r = net.forward(voice, sp_id)

            voice_loss = loss_m(voice, mel_output)
            rhythm_loss = loss_l(rhythm, rhythm_r)
            content_loss = loss_l(content, content_r)

            recon_voice_loss = voice_loss + (config.opt.lambda_r * rhythm_loss) + (config.opt.lambda_c * content_loss)

            pitch_predition_loss = loss_m(pitch_t, pitch_p)
            pitch_embedding_loss = loss_l(pitch_embedding, pitch_embedding_r)

            recon_pitch_loss = pitch_predition_loss + (config.opt.lambda_p * pitch_embedding_loss)

            total_loss = recon_voice_loss + recon_pitch_loss

            optimizer.zero_grad()
            total_loss.backward()
            # recon_loss.backward()
            optimizer.step()
            torch.autograd.set_detect_anomaly(True)


            # if step % 10 ==0:
                # writer.add_images('mel-spectrogram/voice_target', voice.transpose(1, 2).unsqueeze(dim=1), global_step=batch_id, dataformats='NCHW')
                # writer.add_images('mel-spectrogram/voice_prediction', mel_output.transpose(1, 2).unsqueeze(dim=1), global_step=batch_id, dataformats='NCHW')

        voice_t = librosa.display.specshow(voice[0].cpu().numpy(), sr=16000)
        plt.savefig("./fig/voice_t/{}_s".format(curr_epoch+1))
        voice_p = librosa.display.specshow(mel_output[0].cpu().detach().numpy(), sr=16000)
        plt.savefig("./fig/voice_p/{}_s".format(curr_epoch+1))

        scheduler.step()
        writer.close()
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, config.opt.save_path +"{}.pth".format(curr_epoch+1))
        print("voice_loss : {:.5f} rhythm_loss : {:.5f} content_loss : {:.5f} recon_voice_loss : {:.5f}\n".format(voice_loss, rhythm_loss, content_loss, recon_voice_loss))
        print("pitch_predition_loss : {:.5f} pitch_embedding_loss : {:.5f} \n".format(pitch_predition_loss, pitch_embedding_loss))
        print("total_loss : %.5lf\n" % total_loss)
        print("Learning rate : %.9f\n" % optimizer.param_groups[0]['lr'])
