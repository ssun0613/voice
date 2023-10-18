import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.generator import generator as G
from model.discriminator import discriminator as D

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
    generator_r = G(opt, device).to(device)
    generator_f = G(opt, device).to(device)
    discriminator_r = D(opt, device).to(device)
    discriminator_f = D(opt, device).to(device)
    # -------------------------------------------- setup dataload --------------------------------------------
    if not opt.debugging:
        from ssun.Voice_trans.data.dataload_dacon import get_loader
    else:
        from Voice_trans.data.dataload_dacon import get_loader
    dataload = get_loader(opt)
    # -------------------------------------------- setup optimizer --------------------------------------------
    g_r = list(generator_r.parameters())
    g_f = list(generator_f.parameters())
    d_r = list(discriminator_r.parameters())
    d_f = list(discriminator_f.parameters())
    if opt.optimizer_name == 'Adam':
        optimizer_g = torch.optim.Adam((g_r+g_f), lr=opt.lr)
        optimizer_d = torch.optim.Adam((d_r+d_f), lr=opt.lr)
    elif opt.optimizer_name == 'RMSprop':
        optimizer_g = torch.optim.RMSprop((g_r+g_f), lr=opt.lr)
        optimizer_d = torch.optim.RMSprop((d_r+d_f), lr=opt.lr)
    else:
        optimizer_g = None
        optimizer_d = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))
    # -------------------------------------------- setup scheduler --------------------------------------------
    if opt.scheduler_name == 'cycliclr':
        scheduler_g = lr_scheduler.CyclicLR(optimizer_g, base_lr=1e-9, max_lr=opt.lr, cycle_momentum=False, step_size_up=3, step_size_down=17, mode='triangular2')
        scheduler_d = lr_scheduler.CyclicLR(optimizer_d, base_lr=1e-9, max_lr=opt.lr, cycle_momentum=False, step_size_up=3, step_size_down=17, mode='triangular2')
    elif opt.scheduler_name == 'cosine':
        scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=10, eta_min=1e-9)
        scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=10, eta_min=1e-9)
    else:
        scheduler_g = None
        scheduler_d = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))

    return device, generator_r, generator_f, discriminator_r, discriminator_f, dataload, optimizer_g, optimizer_d, scheduler_g, scheduler_d

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

    total_loss = recon_voice_loss + recon_pitch_loss

    return voice_loss, rhythm_loss, content_loss, recon_voice_loss, pitch_predition_loss, pitch_embedding_loss, recon_pitch_loss, total_loss

def compute_D_loss(mel_dr_rr, mel_df_rf, mel_dr_fr, mel_df_ff):
    loss_d_r_r = torch.mean((1 - mel_dr_rr) ** 2)
    loss_d_r_f = torch.mean((0 - mel_dr_fr) ** 2)

    loss_dr = (loss_d_r_r + loss_d_r_f) / 2.0

    loss_d_f_r = torch.mean((1 - mel_df_rf) ** 2)
    loss_d_f_f = torch.mean((0 - mel_df_ff) ** 2)

    loss_df = (loss_d_f_r + loss_d_f_f) / 2.0

    total_loss_d = (loss_dr + loss_df) / 2.0
    return total_loss_d
def tensorboard_draw(voice, mel_output, voice_loss, rhythm_loss, content_loss, recon_voice_loss, pitch_predition_loss, pitch_embedding_loss, recon_pitch_loss, total_loss, global_step):
    writer.add_scalar("encoder/voice_loss", voice_loss, global_step)
    writer.add_scalar("encoder/rhythm_loss", rhythm_loss, global_step)
    writer.add_scalar("encoder/content_loss", content_loss, global_step)
    writer.add_scalar("encoder/recon_voice_loss", recon_voice_loss, global_step)

    writer.add_scalar("pitch_preditor/pitch_predition_loss", pitch_predition_loss, global_step)
    writer.add_scalar("pitch_preditor/pitch_embedding_loss", pitch_embedding_loss, global_step)
    writer.add_scalar("pitch_preditor/recon_pitch_loss", recon_pitch_loss, global_step)

    writer.add_scalar("total_loss", total_loss, global_step)

    spectrogram_target = []
    spectrogram_prediction = []

    for i in range(voice.shape[0]):
        target_spectogram = (voice[i].unsqueeze(dim=0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        target_spectogram = cv2.applyColorMap(target_spectogram, cv2.COLORMAP_JET)
        spectrogram_target.append(target_spectogram)

        prediction_spectogram = (mel_output[i].unsqueeze(dim=0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
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
    device, generator_r, generator_f, discriminator_r, discriminator_f, dataload, optimizer_g, optimizer_d, scheduler_g, scheduler_d = setup(config.opt)
    setproctitle(config.opt.network_name)

    global_step = 0
    for curr_epoch in range(config.opt.epochs):
        print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch+1))
        for batch_id, data in enumerate(dataload, 1):
            global_step+=1

            # data : melsp, mfcc, pitch, len_org, sp_id
            voice = data['melsp'].to(device)
            pitch_t = data['pitch'].to(device)
            sp_id = data['sp_id'].to(device)

            # ---------------- generator train ----------------
            mel_output_g_r, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r = generator_r.forward(voice, sp_id)

            voice_loss, rhythm_loss, content_loss, recon_voice_loss, pitch_predition_loss, pitch_embedding_loss, recon_pitch_loss, total_loss_g \
                = compute_G_loss(voice, pitch_t, mel_output_g_r, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r)

            optimizer_g.zero_grad()
            total_loss_g.backward()
            optimizer_g.step()
            torch.autograd.set_detect_anomaly(True)

            # ---------------- discriminator train ----------------
            mel_dr_rr = discriminator_r.forward(voice)
            mel_df_rf = discriminator_f.forward(mel_output_g_r)

            mel_gr_f, _, _, _, _, _, _, _ = generator_r.forward(voice, sp_id)
            mel_gf_r, _, _, _, _, _, _, _ = generator_f.forward(mel_output_g_r, sp_id)

            mel_dr_fr = discriminator_r.forward(mel_gf_r)
            mel_df_ff = discriminator_f.forward(mel_gr_f)

            if batch_id % 100 == 0:
                total_loss_d = compute_D_loss(mel_dr_rr, mel_df_rf, mel_dr_fr, mel_df_ff)

                optimizer_d.zero_grad()
                total_loss_d.backward()
                optimizer_d.step()
                torch.autograd.set_detect_anomaly(True)

            # -----------------------------------------------------

            if batch_id % 500 == 0:
                tensorboard_draw(voice, mel_output_g_r, voice_loss, rhythm_loss, content_loss, recon_voice_loss, pitch_predition_loss, pitch_embedding_loss, recon_pitch_loss, total_loss_g, global_step)

        scheduler_g.step()
        scheduler_d.step()
        writer.close()

        torch.save({'generator_r': generator_r.state_dict(), 'generator_f': generator_f.state_dict(), 'discriminator_r': discriminator_r.state_dict(), 'discriminator_f': discriminator_f.state_dict(), 'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict()}, config.opt.save_path +"{}.pth".format(curr_epoch+1))

        print("total_loss_g : %.5lf\n" % total_loss_g)
        print("total_loss_d : %.5lf\n" % total_loss_d)
