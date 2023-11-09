import os,sys
sys.path.append("..")

import torch
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.generator import generator as G
from model.discriminator_star import Discriminator as D
from model.encoder import InterpLnr

from torch.utils.tensorboard import SummaryWriter
from functions.load_network import load_networks, init_weights
from functions.loss_function import compute_G_loss, compute_D_loss
from functions.draw_function import tensorboard_draw
from functions.etc_fcn import quantize_f0_torch

import matplotlib
matplotlib.use("Agg")

 # python train.py --debugging True --batch_size 2 --epochs 100000 --tensor_name no_discriminator_encoder_ch --checkpoint_name no_discriminator_encoder_ch

def setup(opt):
    #-------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        if not opt.debugging:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # -------------------------------------------- setup dataload --------------------------------------------
    if not opt.debugging:
        from ssun.Voice_trans.data.dataload_dacon import get_loader
    else:
        from Voice_trans.data.dataload_dacon import get_loader
    dataload = get_loader(opt)
    # -------------------------------------------- setup network --------------------------------------------
    generator = G(opt, device).to(device)
    discriminator = D().to(device)  # star_gan

    if opt.continue_train:
        generator = load_networks(generator, opt.checkpoint_load_num, device, net_name='generator', weight_path= "/storage/mskim/checkpoint/{}/".format(opt.checkpoint_name))
        discriminator = load_networks(discriminator, opt.checkpoint_load_num, device, net_name='discriminator', weight_path= "/storage/mskim/checkpoint/{}/".format(opt.checkpoint_name))
    # -------------------------------------------- setup optimizer --------------------------------------------
    if opt.optimizer_name == 'Adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
    elif opt.optimizer_name == 'RMSprop':
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
        optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
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

    return device, generator, discriminator, dataload, optimizer_g, optimizer_d, scheduler_g, scheduler_d


if __name__ == "__main__":
    config = Config()
    config.print_options()

    setproctitle(config.opt.network_name)
    # torch.cuda.set_device(int(config.opt.gpu_id))
    writer = SummaryWriter('./runs/{}'.format(config.opt.tensor_name))
    device, generator, discriminator, dataload, optimizer_g, optimizer_d, scheduler_g, scheduler_d = setup(config.opt)
    print(device)

    global_step = 0
    for curr_epoch in range(config.opt.epochs):
        print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch+1))
        for batch_id, data in enumerate(dataload, 1):
            global_step+=1

            mel_in = data['melsp'].to(device)
            pitch_t = data['pitch'].to(device)
            len_org = data['len_org'].to(device)
            sp_id = data['sp_id'].to(device)

            # ---------------- generator train ----------------

            mel_out, pitch_p, rhythm, content, rhythm_r, content_r, pitch_p_r = generator.forward(mel_in, sp_id)

            d_r_mel_in = discriminator.forward(mel_in)
            d_r_mel_out = discriminator.forward(mel_out)

            # ---------------- generator loss compute ----------------

            recon_voice_loss, recon_pitch_loss, total_loss_g = compute_G_loss(config.opt, mel_in, pitch_t, mel_out, pitch_p, rhythm, content, rhythm_r, content_r, pitch_p_r, d_r_mel_out)
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss_g.backward(retain_graph=True)
            optimizer_g.step()

            if batch_id % 500 == 0:
                # ---------------- discriminator train ----------------
                d_r_mel_in = discriminator.forward(mel_in)
                d_r_mel_out = discriminator.forward(mel_out.detach())

                # ---------------- discriminator loss compute ----------------

                total_loss_d = compute_D_loss(d_r_mel_in, d_r_mel_out)
                total_loss_d = total_loss_d * 100

                optimizer_d.zero_grad()
                total_loss_d.backward()
                optimizer_d.step()

                # -----------------------------------------------------

            if batch_id % 5 == 0:
                # tensorboard_draw(writer, mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_d, global_step)
                tensorboard_draw(writer, mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_g, global_step)

        scheduler_g.step()
        # scheduler_d.step()
        writer.close()

        print("total_loss_g : %.5lf\n" % total_loss_g)
        # print("total_loss_d : %.5lf\n" % total_loss_d)

        if curr_epoch % 50 == 0 :
            os.makedirs(("/storage/mskim/checkpoint/{}".format(config.opt.checkpoint_name)), exist_ok=True)
            torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict()}, "/storage/mskim/checkpoint/{}/{}.pth".format(config.opt.checkpoint_name, curr_epoch + 1))

