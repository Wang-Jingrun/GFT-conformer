import time, math
import torch.nn as nn
import torch.nn.functional as F
from .conformer import *
from utils.gsp import *


class Encoder(nn.Module):
    """
    Class of upsample block
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        conved = self.conv(x)
        normed = self.bn(conved)
        acted = self.PReLU(normed)

        return acted


class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.convt = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=self.filter_size, stride=self.stride_size,
                                        output_padding=self.output_padding, padding=self.padding)

        if not self.last_layer:
            self.bn = nn.BatchNorm2d(num_features=self.out_channels)
            self.PReLU = nn.PReLU()

    def forward(self, x):

        conved = self.convt(x)

        if not self.last_layer:
            normed = self.bn(conved)
            output = self.PReLU(normed)
        else:
            output = conved

        return output


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

    def forward(self, x_in):
        b, c, f, t = x_in.size()
        x_t = x_in.permute(0, 2, 3, 1).contiguous().view(b*f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 2, 1)
        return x_f
        
class GFT_conformer(nn.Module):
    def __init__(self, device='cpu',
                 n_f=512, win_l=None, n_s=128,
                 kernel_size=5,
                 kernel_num=(64, 64, 64, 64),
                 U_path='', training=True
                 ):
        super(GFT_conformer, self).__init__()

        self.kernel_size = kernel_size
        self.kernel_num = (1,) + kernel_num

        self.n_f = n_f
        self.device = device
        self.gsp = GSP(n_f=n_f, win_l=win_l, n_s=n_s, device=device, U_path=U_path, training=training)
        
        self.k = nn.Parameter(torch.ones(n_f))
        self.c = nn.Parameter(torch.ones(n_f))
        self.b = nn.Parameter(torch.ones(n_f)*0.01)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(Encoder(in_channels=self.kernel_num[idx], out_channels=self.kernel_num[idx + 1],
                                        filter_size=(self.kernel_size, 2), stride_size=(2, 1), padding=(2, 0)))

        self.enhance = TSCB(num_channel=self.kernel_num[-1])
        self.enhance1 = TSCB(num_channel=self.kernel_num[-1])
        self.enhance2 = TSCB(num_channel=self.kernel_num[-1])
        self.enhance3 = TSCB(num_channel=self.kernel_num[-1])


        for idx in range(len(self.kernel_num) - 1, 0, -1):
            last_layer = False if idx != 1 else True
            self.decoder.append(Decoder(in_channels=self.kernel_num[idx] * 2, out_channels=self.kernel_num[idx - 1],
                                        filter_size=(self.kernel_size, 2), stride_size=(2, 1), padding=(2, 0),
                                        output_padding=(1, 0), last_layer=last_layer))

    def forward(self, x):
        gft = self.gsp.ST_GFT(x)
        out = gft.unsqueeze(1)

        # encoder
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = F.pad(out, [1, 0, 0, 0])
            out = encoder(out)
            encoder_out.append(out)

        # enhance
        out = self.enhance(out)
        out = self.enhance1(out)
        out = self.enhance2(out)
        out = self.enhance3(out)
        # out = self.enhance4(out)

        # decoder
        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]

        mask = out.squeeze(1)
        mask = (self.k * torch.tanh(self.c * mask.permute(0, 2, 1)) + self.b).permute(0, 2, 1)
        enhance = mask * gft

        out_wav = self.gsp.iST_GFT(enhance)
        out_wav = out_wav[:, :x.shape[1]]
        out_wav = torch.clamp_(out_wav, -1, 1)
        return out_wav, enhance




