import torch
import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np
#from rfeb_net import rfb_net
import torch.nn.functional as F
import math
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        #print(m.kernel_size, m.out_channels)
        m.weight.data.normal_(0.0,0.5*math.sqrt(2./n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d')!=-1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear')!=-1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'single_stage':
        netG = one_stage()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init_G)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    # print('Total number of parameters: %d' % num_params)
    print("net Total params: %.2fM" % (num_params / 1e6))


class rfb_net(nn.Module):
    def __init__(self, inc):
        super(rfb_net, self).__init__()
        self.one = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=1, padding=0),
                                 nn.ReLU(),
                                 nn.Conv2d(inc*2, inc, kernel_size=(3, 3), padding=(1, 1)))

        self.two = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=1, padding=0),
                                 nn.ReLU(),
                                    nn.Conv2d(inc*2, inc*2, kernel_size=(1, 3), padding=(0, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(inc*2, inc*2, kernel_size=(1, 3), padding=(0, 1)),
                                    nn.ReLU(),
                                    #nn.Conv2d(inc * 2, inc *2, kernel_size=(3, 3), padding=(1, 1), dilation=1),
                                    #nn.ReLU(),
                                    nn.Conv2d(inc*2, inc, kernel_size=(3, 3), padding=(2, 2), dilation=2)
                                )

        self.three = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(inc * 2, inc * 2, kernel_size=(3, 1), padding=(1, 0)),
                                   nn.ReLU(),
                                   nn.Conv2d(inc * 2, inc * 2, kernel_size=(3, 1), padding=(1, 0)),
                                   nn.ReLU(),
                                   #nn.Conv2d(inc * 2, inc*2, kernel_size=(3, 3), padding=(1, 1), dilation=1),
                                   #nn.ReLU(),
                                   nn.Conv2d(inc * 2, inc, kernel_size=(3, 3), padding=(2, 2), dilation=2)
                                   )
        self.five = nn.Sequential(
                                    nn.Conv2d(inc, inc * 2, kernel_size=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(inc * 2, inc * 2, kernel_size=(3, 1), padding=(1, 0)),
                                    nn.ReLU(),
                                    nn.Conv2d(inc * 2, inc * 2, kernel_size=(3, 1), padding=(1, 0)),
                                    nn.ReLU(),
                                    nn.Conv2d(inc * 2, inc * 2, kernel_size=(1, 3), padding=(0, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(inc * 2, inc * 2, kernel_size=(1, 3), padding=(0, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(inc * 2, inc, kernel_size=(3, 3), padding=(1, 1), dilation=1),
                                    #nn.Conv2d(inc, inc, kernel_size=(3, 3), padding=(2, 2), dilation=2),
                                    #nn.Conv2d(inc, inc, kernel_size=(3, 3), stride=1, padding=(5, 5), dilation=5),
                                    nn.BatchNorm2d(inc)
                                    )

        self.four = nn.Conv2d(inc, inc, kernel_size=1)

        self.concatconv = nn.Conv2d(inc * 4, inc, kernel_size=1)

    def forward(self, x):
        feature = []
        feature.append(self.one(x))
        feature.append(self.two(x))
        feature.append(self.three(x))
        feature.append(self.four(x))
        feature.append(self.five(x))
        temp = torch.cat((feature[0], feature[1], feature[2], feature[4]), dim=1)
        temp = self.concatconv(temp)#feature[0] + feature[1] + feature[2])

        out = 0.5 * temp + feature[3]
        return out

class RFEB_Relu(nn.Module):
    def __init__(self, inc):
        super(RFEB_Relu, self).__init__()
        self.RFEB1 = rfb_net(inc)
        #self.RFEB2 = rfb_net(inc)
        self.c = nn.Conv2d(inc, inc, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        a = self.RFEB1(x)
        a = F.relu(a)
        a = self.c(a)
        a = F.relu(a)
        return a + x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
'''
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
'''

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
        scale = 4
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

        res = torch.cat(MSRB_out, 1)
        x = self.Edge_Net_tail(res)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channel, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h, self.pool_c = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1)), nn.AdaptiveAvgPool2d(1)
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, in_channels, kernel_size=(1,1), stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, in_channels, kernel_size=(1,1), stride=1, padding=0)
        self.conv4 = nn.Conv2d(temp_c, in_channels, kernel_size=(1,1), stride=1, padding=0)

        self.fusion_layer = nn.Conv2d(in_channels, out_channel, 1, padding=0, bias=False)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        # x_h:[n,c,h,1]    x_w:[n,c,1,w]   x_c:[n,c,1,1]
        x_h, x_w, x_c = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.pool_c(x)

        # x_cat:[n,c,h+w+1,1]
        x_cat = torch.cat([x_h, x_w, x_c], dim=2)

        out = self.act1(self.bn1(self.conv1(x_cat)))

        # x_h:[n,c,h,1]   x_w:[n,c,w,1]   x_c:[n,c,1,1]
        x_h, x_w, x_c = torch.split(out, [H, W, 1], dim=2)
        # x_w:[n,c,1,w]
        x_w = x_w.permute(0, 1, 3, 2)
        # out_h:[n,c,h,1]
        out_h = torch.sigmoid(self.conv2(x_h))
        # out_w:[n,c,1,w]
        out_w = torch.sigmoid(self.conv3(x_w))
        # out_c:[n,c,1,1]
        out_c = torch.sigmoid(self.conv4(x_c))
        short = short * out_w * out_h * out_c
        short = self.fusion_layer(short)
        return short

class CAL(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16, bias=False):
        super(CAL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                                    nn.Conv2d(inchannel, inchannel // reduction, 1, padding=0, bias=bias),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inchannel // reduction, inchannel, 1, padding=0, bias=bias),
                                    nn.Sigmoid()
                                    )
        self.fusion_layer = nn.Conv2d(inchannel, outchannel, 1, padding=0, bias=bias)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        x = x * y
        x = self.fusion_layer(x)
        return x

class encoder_decoder(nn.Module):
    def __init__(self):
        super(encoder_decoder, self).__init__()
        self.ec_RFEB1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),#RFEB_Relu(32)
                                      #nn.ReLU(),
                                      RFEB_Relu(32),
                                      #RFEB_Relu(32),
                                      #nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
                                      #nn.ReLU()
                                      )
        self.ec_RFEB2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                                      RFEB_Relu(64),
                                      RFEB_Relu(64),
                                      #nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
                                      #nn.ReLU()
                                      )
        self.ec_RFEB3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
                                      #nn.ReLU(),
                                      RFEB_Relu(128),
                                      RFEB_Relu(128),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
                                      #nn.ReLU()
                                      )

        self.de_RFEB1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),#RFEB_Relu(128)
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),#RFEB_Relu(128)
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
                                      #nn.ReLU()
                                      )

        self.de_RFEB2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                      #nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
                                      #nn.ReLU()
                                      )

        self.de_REFB3= nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                     #nn.ReLU(),
                                     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
                                     #nn.ReLU()
                                     )


        self.edgeup12 = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=2, padding=0)
        self.edgeup23 = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=2, padding=0)
        self.cal33 = CAL(131, 128)
        self.cal23 = CAL(259, 128)
        self.cal12 = CAL(131, 64)

        # self.coo33 = CoordAttention(131, 128)
        # self.coo23 = CoordAttention(259, 128)
        # self.coo12 = CoordAttention(131, 64)
        


    def forward(self, x, edge): # edge
        ec_one = self.ec_RFEB1(x)
        ec_two = self.ec_RFEB2(ec_one)
        ec_three = self.ec_RFEB3(ec_two)

        edge_12 = self.edgeup12(edge)
        edge_23 = self.edgeup23(edge_12)
         
        ec_three = torch.cat((ec_three, edge_23), dim=1)
        ec_three = self.cal33(ec_three)
        de_one = self.de_RFEB1(ec_three)

        de_one = torch.cat((de_one, ec_two, edge_23), dim=1) #
        de_one = self.cal23(de_one)
        de_two = self.de_RFEB2(de_one)

        de_two = torch.cat((de_two, ec_one, edge_12), dim=1) #
        de_two = self.cal12(de_two)
        de_three = self.de_REFB3(de_two)

        return de_three


class one_stage(nn.Module):
    def __init__(self):
        super(one_stage, self).__init__()

        kernel_x = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)#.to(device='cuda')
        # print("kernel_x size: ", kernel_x.size())
        #
        kernel_y = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)#.to(device='cuda')
        # print("kernel_y size: ", kernel_y.size())
        #
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        ###############################
        self.head1 = nn.Conv2d(3, 3, kernel_size=(3,3), stride=1, padding=(1,1))
        self.head2 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=(1,1))
        self.body = encoder_decoder()
        # self.body = scale_atten_m()

        self.feedback = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1,1))

        self.end1 = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=1, padding=(1,1))
        self.end2 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=1, padding=(1,1))

        self.soft_edge = Edge_Net()#.requires_grad_(requires_grad=False)
        self.cal = CAL(6, 3, 1)

    def forward(self, x):
        edge = self.soft_edge(x)#soft_divergence(x)#soft_edge(x)
        x = torch.cat((x, edge), dim=1)
        x = self.cal(x)
        
        result = self.head1(x)
        result = F.relu(result)
        result = self.head2(result)
        sharp_image = None

        for i in range(3):
            ## 经过编码解码器
            sharp_image = self.body(result, edge) #

            ## 更新头部张量result
            loop = self.feedback(sharp_image)
            w = torch.sigmoid(loop)
            result = w * result + result
            result = result + sharp_image
            ## 循环结果，准备加强原图

        sharp_image = self.end1(result)
        sharp_image = self.end2(sharp_image)
        return sharp_image, edge

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2

        # 下采样
        # for i in range(n_downsampling): # [0,1]
        # 	mult = 2**i
        #
        # 	model += [
        # 		nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        # 		norm_layer(ngf * mult * 2),
        # 		nn.ReLU(True)
        # 	]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        # for i in range(n_downsampling):
        # 	mult = 2**(n_downsampling - i)
        #
        # 	model += [
        # 		nn.ConvTranspose2d(
        # 			ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
        # 			padding=1, output_padding=1, bias=use_bias),
        # 		norm_layer(int(ngf * mult / 2)),
        # 		nn.ReLU(True)
        # 	]
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        padAndConv = {
            'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [
                norm_layer(dim),
                nn.ReLU(True)
            ] + [
                nn.Dropout(0.5)
            ] if use_dropout else [] + padAndConv[padding_type] + [
                norm_layer(dim)
            ]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, gpu_ids=[], use_parallel=True, learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
            self, outer_nc, inner_nc, submodule=None,
            outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            model = [
                dModel,
                submodule,
                uModel
            ]

        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = [
                dModel,
                uModel
            ]

        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            model = [
                dModel,
                submodule,
                uModel
            ]
            model += [nn.Dropout(0.5)] if use_dropout else []


        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

