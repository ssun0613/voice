import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.generator import generator as G
from model.discriminator import Discriminator as D

from torch.utils.tensorboard import SummaryWriter

import io
import numpy as np
import cv2
def setup(opt):
    #-------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        if not opt.debugging:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # -------------------------------------------- setup network --------------------------------------------
    generator = G(opt, device).to(device)
    # -------------------------------------------- setup dataload --------------------------------------------
    if not opt.debugging:
        from ssun.Voice_trans.data.dataload_dacon import get_loader
    else:
        from Voice_trans.data.dataload_dacon import get_loader
    dataload = get_loader(opt)
    # -------------------------------------------- setup optimizer --------------------------------------------
    if opt.optimizer_name == 'Adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr)

    elif opt.optimizer_name == 'RMSprop':
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    else:
        optimizer_g = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))
    # -------------------------------------------- setup scheduler --------------------------------------------
    if opt.scheduler_name == 'cycliclr':
        scheduler_g = lr_scheduler.CyclicLR(optimizer_g, base_lr=1e-9, max_lr=opt.lr, cycle_momentum=False, step_size_up=3, step_size_down=17, mode='triangular2')
    elif opt.scheduler_name == 'cosine':
        scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=10, eta_min=1e-9)
    else:
        scheduler_g = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))

    return device, generator, dataload, optimizer_g, scheduler_g
def compute_G_loss(voice, pitch_t, mel_output, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r):
    loss_m = nn.MSELoss(reduction='sum')
    loss_l = nn.L1Loss(reduction='sum')

    voice_loss = loss_m(voice, mel_output)
    rhythm_loss = loss_l(rhythm, rhythm_r)
    content_loss = loss_l(content, content_r)

    recon_voice_loss = voice_loss + (config.opt.lambda_r * rhythm_loss) + (config.opt.lambda_c * content_loss)

    pitch_predition_loss = loss_m(pitch_t, pitch_p)
    pitch_embedding_loss = loss_l(pitch_embedding, pitch_embedding_r)

    recon_pitch_loss = pitch_predition_loss + (config.opt.lambda_p * pitch_embedding_loss)

    total_loss_g = recon_voice_loss + recon_pitch_loss

    return recon_voice_loss, recon_pitch_loss, total_loss_g
def tensorboard_draw(mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, global_step):
    writer.add_scalar("loss/recon_voice_loss", recon_voice_loss, global_step)
    writer.add_scalar("loss/recon_pitch_loss", recon_pitch_loss, global_step)
    writer.add_scalar("loss/total_loss_g", total_loss_g, global_step)

    spectrogram_target = []
    spectrogram_prediction = []

    for i in range(mel_in.shape[0]):
        target_spectogram = (mel_in[i].unsqueeze(dim=0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        target_spectogram = cv2.applyColorMap(target_spectogram, cv2.COLORMAP_JET)
        spectrogram_target.append(target_spectogram)

        prediction_spectogram = (mel_out[i].unsqueeze(dim=0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        prediction_spectogram = cv2.applyColorMap(prediction_spectogram, cv2.COLORMAP_JET)
        spectrogram_prediction.append(prediction_spectogram)

    spectrogram_target = np.array(spectrogram_target)
    spectrogram_prediction = np.array(spectrogram_prediction)
    writer.add_images('mel-spectrogram/voice_target', spectrogram_target, global_step, dataformats='NHWC')
    writer.add_images('mel-spectrogram/voice_prediction', spectrogram_prediction, global_step, dataformats='NHWC')


if __name__ == "__main__":
    config = Config()
    writer = SummaryWriter()
    config.print_options()
    torch.cuda.set_device(int(config.opt.gpu_id))
    device, generator, dataload, optimizer_g, scheduler_g = setup(config.opt)
    setproctitle(config.opt.network_name)

    global_step = 0
    for curr_epoch in range(config.opt.epochs):
        print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch+1))
        for batch_id, data in enumerate(dataload, 1):
            global_step+=1

            mel_in = data['melsp'].to(device)
            pitch_t = data['pitch'].to(device)
            sp_id = data['sp_id'].to(device)

            # ---------------- generator train ----------------

            mel_out, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r = generator.forward(mel_in, sp_id)

            recon_voice_loss, recon_pitch_loss, total_loss_g = compute_G_loss(mel_in, pitch_t, mel_out, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r)

            if batch_id % 100 == 0:
                tensorboard_draw(mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, global_step)

            optimizer_g.zero_grad()
            total_loss_g.backward(retain_graph=True)
            optimizer_g.step()

        scheduler_g.step()
        writer.close()

        print("total_loss_g : %.5lf\n" % total_loss_g)
