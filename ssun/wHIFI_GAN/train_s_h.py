import os, sys, json, matplotlib, warnings, librosa
sys.path.append("..")
matplotlib.use("Agg")
warnings.simplefilter('ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from setproctitle import *
from scipy.io.wavfile import write

from config import Config
from model.speechsplit_ssun import speechsplit, InterpLnr
from model.hifigan import Generator

from torch.utils.tensorboard import SummaryWriter
from functions.load_network import load_networks, init_weights
from functions.etc_fcn import quantize_f0_torch

from hparams import hparams
from utils import AttrDict,build_env, load_checkpoint, mel_spectrogram


def setup(opt):
    #-------------------------------------------- setup device --------------------------------------------
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
        from ssun.wHIFI_GAN.data.dataload_remove_noise import get_loader
    else:
        from wHIFI_GAN.data.dataload_remove_noise import get_loader
    dataload = get_loader(opt)

    # -------------------------------------------- setup network --------------------------------------------
    generator = speechsplit(hparams).to(device)
    if opt.continue_train:
        generator = load_networks(generator, opt.checkpoint_name, device, net_name='generator', weight_path= "/storage/mskim/checkpoint/")

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

def setup_hifi(device):
    hifi_gan_checkpoint = "/storage/mskim/hifigan_pre/generator_v1"
    state_dict_g = load_checkpoint(hifi_gan_checkpoint, device)

    with open("/storage/mskim/hifigan_pre/config.json") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env("/storage/mskim/hifigan_pre/config.json", 'config.json', "/storage/mskim/hifigan_pre")

    hifi_generator = Generator(h).to(device)
    hifi_generator.load_state_dict(state_dict_g['generator'])

    return hifi_generator, h

def tensorboard_draw(writer, mel_in, mel_out, g_loss, global_step, y_g_hat, sampling_rate):

    writer.add_scalar("loss/g_loss", g_loss, global_step)

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

    writer.add_audio('audio/voice_prediction', y_g_hat[0], global_step, sampling_rate)

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


if __name__ == "__main__":

    config = Config()
    config.print_options()

    setproctitle(config.opt.network_name)
    torch.cuda.set_device(int(config.opt.gpu_id))
    writer = SummaryWriter('/storage/mskim/tensorboard/{}'.format(config.opt.tensor_name))
    device, generator, dataload, optimizer_g, scheduler_g = setup(config.opt)
    hifi_generator, h = setup_hifi(device)

    print(device)

    try:
        global_step = 0
        for curr_epoch in range(config.opt.epochs):
            print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch+1))
            for batch_id, data in enumerate(dataload, 1):
                global_step+=1

                mel_in = data['melsp'].to(device)
                pitch_t = data['pitch'].to(device)
                len_org = data['len_org'].to(device)
                sp_id = data['sp_id'].to(device)

                x_f0 = torch.cat((mel_in, pitch_t), dim=-1)
                x_f0_intrp = InterpLnr(hparams)(x_f0, len_org)
                f0_org_intrp = quantize_f0_torch(x_f0_intrp[:, :, -1])[0]
                x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)

                # ---------------- mel_spectogram ----------------

                mel_out = generator.forward(x_f0_intrp_org, mel_in, sp_id)

                # -------------------- wav --------------------
                torch.cuda.empty_cache()
                y_g_hat = hifi_generator(mel_out.transpose(2,1).transpose(2,0).to(device))

                # import soundfile as sf
                # sf_path = '/storage/mskim/prediction_audio.wav'
                # sf.write(sf_path, y_g_hat.cpu().detach().numpy(), h.sampling_rate)

                mel_nrw = librosa.amplitude_to_db(np.abs(librosa.stft(mel_in[0].cpu().detach().numpy())), ref=np.max)
                mel_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_g_hat[0].cpu().detach().numpy())), ref=np.max)
                plt.subplot(2, 1, 1)
                plt.imshow(mel_nrw[::-1, :])
                plt.subplot(2, 1, 2)
                plt.imshow(mel_orig[::-1, :])

                # ---------------- generator loss compute ----------------

                g_loss = nn.MSELoss(reduction='mean')(mel_in, mel_out)

                optimizer_g.zero_grad()
                g_loss.backward(retain_graph=True)
                optimizer_g.step()


            if curr_epoch % 1000 == 0 :
                tensorboard_draw(writer, mel_in, mel_out, g_loss, global_step, y_g_hat, h.sampling_rate)

            scheduler_g.step()
            writer.close()

            print("total_loss_g : %.5lf\n" % g_loss)

            if curr_epoch % 1000 == 0 and not config.opt.continue_train:
                torch.save({'generator': generator.state_dict(), 'optimizer_g': optimizer_g.state_dict()}, "/storage/mskim/checkpoint/{}.pth".format(config.opt.checkpoint_name))


    except Exception as e:
        print(e)
