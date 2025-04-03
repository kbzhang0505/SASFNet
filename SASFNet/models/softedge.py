mport torch.nn as nn
import torch


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MSRB(nn.Module):
    def __init__(self, conv=default_conv):
        super(MSRB, self).__init__()

        n_feats = 64
        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output

class Edge_Net(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(Edge_Net, self).__init__()
        
        kernel_size = 3
        act = nn.ReLU(True)
        n_blocks = 5
        self.n_blocks = n_blocks
        
        modules_head = [conv(3, n_feats, kernel_size)]

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB())

        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            #Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.Edge_Net_head(x)
        res = x

        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1)
        x = self.Edge_Net_tail(res)
        return x 