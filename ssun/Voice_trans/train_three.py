import sys
sys.path.append("..")

import torch
import torch.nn as nn
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

    if not opt.debugging:
        from ssun.Voice_trans.model.Er_Ec import Er_Ec
        from ssun.Voice_trans.model.decoder_s import Decoder_s as Ds
        # from ssun.Voice_trans.model.decoder_f import Decoder_f as Df
        from ssun.Voice_trans.model.pitch_predictor import pitch_predictor as P
    else:
        from Voice_trans.model.Er_Ec import Er_Ec
        from Voice_trans.model.decoder_s import Decoder_s as Ds
        # from ssun.Voice_trans.model.decoder_f import Decoder_f as Df
        from Voice_trans.model.pitch_predictor import pitch_predictor as P

    net_r_c = Er_Ec(opt).to(device)
    net_d = Ds().to(device)
    net_p = P().to(device)

    # -------------------------------------------- setup dataload --------------------------------------------

    if not opt.debugging:
        from ssun.Voice_trans.data.dataload_dacon import get_loader
    else:
        from Voice_trans.data.dataload_dacon import get_loader
    dataload = get_loader(opt)

    # -------------------------------------------- setup optimizer --------------------------------------------

    if opt.optimizer_name == 'Adam':
        optimizer_r_c = torch.optim.Adam(net_r_c.parameters(), lr=opt.lr)
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=opt.lr)
        optimizer_p = torch.optim.Adam(net_p.parameters(), lr=opt.lr)
    elif opt.optimizer_name == 'RMSprop':
        optimizer_r_c = torch.optim.RMSprop(net_r_c.parameters(), lr=opt.lr)
        optimizer_d = torch.optim.RMSprop(net_d.parameters(), lr=opt.lr)
        optimizer_p = torch.optim.RMSprop(net_p.parameters(), lr=opt.lr)
    else:
        optimizer = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))

    # -------------------------------------------- setup scheduler --------------------------------------------

    if opt.scheduler_name == 'cosine':
        scheduler_r_c = lr_scheduler.CosineAnnealingLR(optimizer_r_c, T_max=10, eta_min=1e-9)
        scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=10, eta_min=1e-9)
        scheduler_p = lr_scheduler.CosineAnnealingLR(optimizer_p, T_max=10, eta_min=1e-9)
    else:
        scheduler = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))

    return device, net_r_c, net_d, net_p, dataload, optimizer_r_c, optimizer_d, optimizer_p, scheduler_r_c, scheduler_d, scheduler_p

if __name__ == "__main__":
    config = Config()
    config.print_options()
    device, net_r_c, net_d, net_p, dataload, optimizer_r_c, optimizer_d, optimizer_p, scheduler_r_c, scheduler_d, scheduler_p = setup(config.opt)
    loss = nn.MSELoss()
    print("loss = nn.MSELoss()\n")

    for curr_epoch in range(config.opt.epochs):
        print("-------------------------- Epoch : {} --------------------------".format(curr_epoch+1))
        for batch_id, data in enumerate(dataload, 1):

            # data : melsp, mfcc, pitch, len_org, sp_id
            voice = data['melsp'].to(device)
            pitch_t = data['pitch'].to(device)
            sp_id = data['sp_id'].to(device)

            rhythm, content, r_c_s = net_r_c.forward(voice, sp_id)

            pitch_p, pitch_embedding = net_p(r_c_s)
            r_c_p = torch.cat((rhythm, content, pitch_embedding), dim=-1)

            mel_output = net_d(r_c_p)

            voice_loss = loss(voice, mel_output)
            pitch_loss = loss(pitch_t, pitch_p)

            optimizer_r_c.zero_grad()
            optimizer_d.zero_grad()
            optimizer_p.zero_grad()

            voice_loss.backward()
            pitch_loss.backward()

            optimizer_r_c.step()
            optimizer_d.step()
            optimizer_p.step()


        scheduler_r_c.step()
        scheduler_d.step()
        scheduler_p.step()
        # torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, config.opt.save_path +"{}.pth".format(curr_epoch+1))
        print("voice_loss : %.5lf\n" % voice_loss)
        print("pitch_loss : %.5lf\n" % pitch_loss)
