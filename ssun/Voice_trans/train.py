import os,sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *

from config import Config
from model.generator import generator as G
from model.discriminator import Discriminator as D
# from model.discriminator_star import Discriminator1 as D

from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

 # python train.py --debugging True --batch_size 2 --epochs 100000 --tensor_name no_discriminator_encoder_ch --checkpoint_name no_discriminator_encoder_ch


def load_networks(net, checkpoint_load_num, device, net_name, weight_path=None):
    load_filename = '{}.pth'.format(checkpoint_load_num)
    if weight_path is None:
        ValueError('Should set the weight_path, which is the path to the folder including weights')
    else:
        load_path = weight_path + load_filename
    net = net
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path, map_location=str(device))
    net.load_state_dict(state_dict['{}'.format(net_name)])

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
        net.load_state_dic(state_dict['{}'.format(net_name)])
    print('load completed {}'.format(net_name))

    return net

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
    discriminator = D(opt, device).to(device) # masked_cycle_gan
    # discriminator = D().to(device)  # star_gan

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
def compute_G_loss(voice, pitch_t, mel_output, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r, d_r_mel_out):
    loss_m = nn.MSELoss(reduction='sum')
    loss_l = nn.L1Loss(reduction='sum')

    voice_loss = loss_m(voice, mel_output)
    rhythm_loss = loss_l(rhythm, rhythm_r)
    content_loss = loss_l(content, content_r)

    recon_voice_loss = voice_loss + (config.opt.lambda_r * rhythm_loss) + (config.opt.lambda_c * content_loss)

    pitch_predition_loss = loss_m(pitch_t, pitch_p)
    pitch_embedding_loss = loss_l(pitch_embedding, pitch_embedding_r)

    recon_pitch_loss = pitch_predition_loss + (config.opt.lambda_p * pitch_embedding_loss)

    # loss_dr_for_g = (1 - torch.mean((0 - d_r_mel_out) ** 2)) * 100
    loss_dr_for_g = 0

    total_loss_g = recon_voice_loss + recon_pitch_loss + loss_dr_for_g

    return recon_voice_loss, recon_pitch_loss, total_loss_g
def compute_D_loss(d_r_mel_in, d_r_mel_out):
    loss_d_r_r = torch.mean((1 - d_r_mel_in) ** 2)
    loss_d_r_f = torch.mean((0 - d_r_mel_out) ** 2)
    loss_dr = (loss_d_r_r + loss_d_r_f) / 2.0

    return loss_dr
def tensorboard_draw(mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_d, global_step):
    writer.add_scalar("loss/recon_voice_loss", recon_voice_loss, global_step)
    writer.add_scalar("loss/recon_pitch_loss", recon_pitch_loss, global_step)
    writer.add_scalar("loss/total_loss_g", total_loss_g, global_step)
    writer.add_scalar("loss/total_loss_d", total_loss_d, global_step)

    # spectrogram_target = []
    # spectrogram_prediction = []
    #
    # for i in range(mel_in.shape[0]):
    #     target_spectogram = (mel_in[i].unsqueeze(dim=0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    #     target_spectogram = cv2.applyColorMap(target_spectogram, cv2.COLORMAP_JET)
    #     spectrogram_target.append(target_spectogram)
    #
    #     prediction_spectogram = (mel_out[i].unsqueeze(dim=0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    #     prediction_spectogram = cv2.applyColorMap(prediction_spectogram, cv2.COLORMAP_JET)
    #     spectrogram_prediction.append(prediction_spectogram)
    #
    # spectrogram_target = np.array(spectrogram_target)
    # spectrogram_prediction = np.array(spectrogram_prediction)
    # writer.add_images('mel-spectrogram/voice_target', spectrogram_target, global_step, dataformats='NHWC')
    # writer.add_images('mel-spectrogram/voice_prediction', spectrogram_prediction, global_step, dataformats='NHWC')

    spectrogram_target = []
    spectrogram_prediction = []

    for i in range(mel_in.shape[0]):
        target_spectogram = (mel_in[i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        target_spectogram = plot_spectrogram(target_spectogram)
        spectrogram_target.append(target_spectogram)

        prediction_spectogram = (mel_out[i].squeeze(0).permute(1, 0).cpu().detach().numpy())
        prediction_spectogram = plot_spectrogram(prediction_spectogram)
        spectrogram_prediction.append(prediction_spectogram)

    writer.add_figure('mel-spectrogram/voice_target', spectrogram_target, global_step)
    writer.add_figure('mel-spectrogram/voice_prediction', spectrogram_prediction, global_step)
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

if __name__ == "__main__":
    config = Config()
    writer = SummaryWriter('./runs/{}'.format(config.opt.tensor_name))
    config.print_options()
    torch.cuda.set_device(int(config.opt.gpu_id))
    device, generator, discriminator, dataload, optimizer_g, optimizer_d, scheduler_g, scheduler_d = setup(config.opt)
    os.makedirs(("/storage/mskim/checkpoint/{}".format(config.opt.checkpoint_name)), exist_ok=True)
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

            d_r_mel_in = discriminator.forward(mel_in)
            d_r_mel_out = discriminator.forward(mel_out)

            # ---------------- generator loss compute ----------------

            recon_voice_loss, recon_pitch_loss, total_loss_g = compute_G_loss(mel_in, pitch_t, mel_out, pitch_p, pitch_embedding, rhythm, content, rhythm_r, content_r, pitch_embedding_r, d_r_mel_out)

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
                # tensorboard_draw(mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_d, global_step)
                tensorboard_draw(mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_g, global_step)

        scheduler_g.step()
        scheduler_d.step()
        writer.close()

        print("total_loss_g : %.5lf\n" % total_loss_g)
        # print("total_loss_d : %.5lf\n" % total_loss_d)

        if curr_epoch % 50 == 0 :
            torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'optimizer_g': optimizer_g.state_dict(), 'optimizer_d': optimizer_d.state_dict()}, "/storage/mskim/checkpoint/{}/{}.pth".format(config.opt.checkpoint_name, curr_epoch + 1))

