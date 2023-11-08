import numpy as np
import torch
import torch.nn as nn

class ConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sd):
        super(ConvGLU1D, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch * 2, ks, stride=sd, padding=(ks - sd) // 2)
        nn.init.xavier_normal_(self.conv1.weight, gain=0.1)
        self.norm1 = nn.InstanceNorm1d(out_ch * 2)
        self.conv1 = nn.utils.weight_norm(self.conv1)
    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h_l, h_g = torch.split(h, h.shape[1] // 2, dim=1)
        h = h_l * torch.sigmoid(h_g)

        return h
class Discriminator(nn.Module):
    # 1D convolutional architecture
    def __init__(self):
        super(Discriminator, self).__init__()
        in_ch = 192
        mid_ch = 80
        dor = 0.1
        self.le1 = ConvGLU1D(in_ch, mid_ch, 9, 1)
        self.le2 = ConvGLU1D(mid_ch, mid_ch, 8, 2)
        self.le3 = ConvGLU1D(mid_ch, mid_ch, 8, 2)
        self.le4 = ConvGLU1D(mid_ch, mid_ch, 5, 1)
        self.le_adv = nn.Conv1d(mid_ch, 20, 5, stride=1, padding=(5 - 1) // 2, bias=False)
        nn.init.xavier_normal_(self.le_adv.weight, gain=0.1)
        self.do1 = nn.Dropout(p=dor)
        self.do2 = nn.Dropout(p=dor)
        self.do3 = nn.Dropout(p=dor)
        self.do4 = nn.Dropout(p=dor)

    def forward(self, x):
        out = self.do1(self.le1(x))
        out = self.do2(self.le2(out))
        out = self.do3(self.le3(out))
        out = self.do4(self.le4(out))
        out_adv = self.le_adv(out)

        return out_adv


if __name__ == '__main__':
    from ssun.Voice_trans.config import Config
    config = Config()
    device = torch.device("cpu")

    np.random.seed(0)
    input = torch.from_numpy(np.random.randn(10, 192, 80)).float()
    discriminator = Discriminator()
    output = discriminator(input)
    print("Discriminator output shape ", output.shape) # torch.Size([10, 20, 20])