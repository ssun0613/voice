import os, sys, json, matplotlib, warnings
sys.path.append("..")
matplotlib.use("Agg")
warnings.simplefilter('ignore', category=FutureWarning)

import torch
from torch.optim import lr_scheduler
from setproctitle import *
from scipy.io.wavfile import write

from config import Config
from model.generator import generator_original as G
from model.hifigan import Generator

from torch.utils.tensorboard import SummaryWriter
from functions.load_network import load_networks, init_weights
from functions.loss_function import compute_G_loss, compute_D_loss
from functions.draw_function import tensorboard_draw
from functions.etc_fcn import quantize_f0_torch

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
    generator = G(opt).to(device)
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

                pitch_t_quan = quantize_f0_torch(pitch_t)[0]

                # ---------------- mel_spectogram ----------------

                mel_out, pitch_p_repeat_quan, rhythm, content, rhythm_r, content_r, pitch_p_r_quan = generator.forward(mel_in, len_org, sp_id)

                # -------------------- wav --------------------
                torch.cuda.empty_cache()
                y_g_hat = hifi_generator(mel_out.transpose(2,1).transpose(2,0).to(device))
                write("/storage/mskim/audio.wav", 22050, y_g_hat[0].cpu().detach().numpy())

                # ---------------- generator loss compute ----------------

                recon_voice_loss, recon_pitch_loss, total_loss_g = compute_G_loss(config.opt, mel_in, pitch_t_quan, mel_out, pitch_p_repeat_quan, rhythm, content, rhythm_r, content_r, pitch_p_r_quan)

                optimizer_g.zero_grad()
                total_loss_g.backward(retain_graph=True)
                optimizer_g.step()


            if curr_epoch % 100 == 0 :
                tensorboard_draw(writer, mel_in, mel_out, recon_voice_loss, recon_pitch_loss, total_loss_g, total_loss_g, global_step, y_g_hat, h.sampling_rate)

            scheduler_g.step()
            writer.close()


            print("total_loss_g : %.5lf\n" % total_loss_g)

            if curr_epoch % 500 == 0 :
                torch.save({'generator': generator.state_dict(), 'optimizer_g': optimizer_g.state_dict()}, "/storage/mskim/checkpoint/{}.pth".format(config.opt.checkpoint_name))

    except Exception as e:
        print(e)
