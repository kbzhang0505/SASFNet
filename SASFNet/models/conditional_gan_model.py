import numpy as np
import cv2
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss
import torch.nn.functional as F
from torchvision import transforms
from util.lookahead import Lookahead

try:
    xrange  # Python2
except NameError:
    xrange = range  # Python 3


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class local_max_gradients():
    def __init__(self):
        self.mse = torch.nn.MSELoss()

        self.kernel_x = [[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]]
        self.kernel_x = torch.FloatTensor(self.kernel_x).unsqueeze(0).unsqueeze(0)

        self.kernel_y = [[-1., -2., 1.],
                         [0., 0., 0.],
                         [1., 2., 1.]]
        self.kernel_y = torch.FloatTensor(self.kernel_y).unsqueeze(0).unsqueeze(0)

        self.weight_x = torch.nn.Parameter(data=self.kernel_x, requires_grad=False).cuda()
        self.weight_y = torch.nn.Parameter(data=self.kernel_y, requires_grad=False).cuda()

    def get_loss(self, x, target):
        x_b = torch.split(x, 30, dim=2)
        t_b = torch.split(target, 30, dim=2)

        block_x = []
        block_t = []
        for i in range(len(x_b)):
            c = torch.split(x_b[i], 40, dim=3)
            c_t = torch.split(t_b[i], 40, dim=3)
            block_x += c
            block_t += c_t

        max_grad_x = self.local_gradient(block_x)
        max_grad_t = self.local_gradient(block_t)
        max_grad_x = torch.Tensor(max_grad_x)
        max_grad_t = torch.Tensor(max_grad_t)
        return self.mse(max_grad_x, max_grad_t)

    def local_gradient(self, block):
        max_grad = []
        for i in block:
            x1 = i[:, 0, :, :]
            x2 = i[:, 1, :, :]
            x3 = i[:, 2, :, :]
            x1_x = F.conv2d(x1.unsqueeze(0), self.weight_x, padding=1)
            x1_y = F.conv2d(x1.unsqueeze(0), self.weight_y, padding=1)
            x2_x = F.conv2d(x2.unsqueeze(0), self.weight_x, padding=1)
            x2_y = F.conv2d(x2.unsqueeze(0), self.weight_y, padding=1)
            x3_x = F.conv2d(x3.unsqueeze(0), self.weight_x, padding=1)
            x3_y = F.conv2d(x3.unsqueeze(0), self.weight_y, padding=1)

            x1_gradient = torch.pow((torch.pow(x1_x, 2) + torch.pow(x1_y, 2)), 0.5)
            x2_gradient = torch.pow((torch.pow(x2_x, 2) + torch.pow(x2_y, 2)), 0.5)
            x3_gradient = torch.pow((torch.pow(x3_x, 2) + torch.pow(x3_y, 2)), 0.5)

            max_grad_x1 = x1_gradient.max()
            max_grad_x2 = x2_gradient.max()
            max_grad_x3 = x3_gradient.max()

            if max_grad_x3 >= max_grad_x2 and max_grad_x3 >= max_grad_x1:
                max_grad.append(max_grad_x3)
            elif max_grad_x2 >= max_grad_x1 and max_grad_x2 >= max_grad_x3:
                max_grad.append(max_grad_x2)
            elif max_grad_x1 >= max_grad_x2 and max_grad_x1 >= max_grad_x3:
                max_grad.append(max_grad_x1)

            '''
            h_x = i.size()[2]
            w_x = i.size()[3]
            r = F.pad(i, (0,1,0,0))[:, :, :, 1:]
            l = F.pad(i, (1,0,0,0))[:,:,:,:w_x]
            t = F.pad(i, (0,0,1,0))[:, :, :h_x, :]
            b = F.pad(i, (0,0,0,1))[:,:,1:,:]
            xgrad = torch.pow(torch.pow((r-l)*0.5,2) + torch.pow((t-b)*0.5,2), 0.5)
            
            #grad_x = F.conv2d(i, self.weight_x)
            #grad_y = F.conv2d(i, self.weight_y)
            #grad = torch.pow((torch.pow(grad_x, 2)+torch.pow(grad_y, 2)), 0.5)
            
            max_grad.append(torch.max(xgrad))
            '''
        return max_grad


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class soft_edge_loss(nn.Module):
    def __init__(self):
        super(soft_edge_loss, self).__init__()
        # 求梯度
        kernel_x = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device='cuda')
        print("kernel_x size: ", kernel_x.size())

        kernel_y = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device='cuda')
        print("kernel_y size: ", kernel_y.size())

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.loss = torch.nn.L1Loss()

    def soft_divergence(self, x):
        # 分通道计算
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]

        # 求梯度
        x1 = x1.unsqueeze(1)
        # print("X1 size: ", x1.size())
        grad_x1 = F.conv2d(x1, self.weight_x, padding=1)
        grad_y1 = F.conv2d(x1, self.weight_y, padding=1)
        m = torch.sqrt((1 + torch.pow(grad_x1, 2) + torch.pow(grad_y1, 2)))
        grad_x1 = grad_x1 / m
        grad_y1 = grad_y1 / m
        grad_x1 = F.conv2d(grad_x1, self.weight_x, padding=1)
        grad_y1 = F.conv2d(grad_y1, self.weight_y, padding=1)
        R_TD = grad_x1 + grad_y1

        x2 = x2.unsqueeze(1)
        grad_x2 = F.conv2d(x2, self.weight_x, padding=1)
        grad_y2 = F.conv2d(x2, self.weight_y, padding=1)
        m = torch.sqrt((1 + torch.pow(grad_x2, 2) + torch.pow(grad_y2, 2)))
        grad_x2 = grad_x2 / m
        grad_y2 = grad_y2 / m
        grad_x2 = F.conv2d(grad_x2, self.weight_x, padding=1)
        grad_y2 = F.conv2d(grad_y2, self.weight_y, padding=1)
        G_TD = grad_x2 + grad_y2

        x3 = x3.unsqueeze(1)
        grad_x3 = F.conv2d(x3, self.weight_x, padding=1)
        grad_y3 = F.conv2d(x3, self.weight_y, padding=1)
        m = torch.sqrt((1 + torch.pow(grad_x3, 2) + torch.pow(grad_y3, 2)))
        grad_x3 = grad_x3 / m
        grad_y3 = grad_y3 / m
        grad_x3 = F.conv2d(grad_x3, self.weight_x, padding=1)
        grad_y3 = F.conv2d(grad_y3, self.weight_y, padding=1)
        B_TD = grad_x3 + grad_y3

        RGB_TD = torch.cat((R_TD, G_TD, B_TD), dim=1)
        return RGB_TD

    def forward(self, x, target):
        # print("x size: ", x.size())
        x = self.soft_divergence(x)
        target = self.soft_divergence(target)
        x = F.relu(x)
        target = F.relu(target)
        res = self.loss(x, target)
        # res = torch.norm((x_d - t_d), p='fro', dim=None, keepdim=False, out=None, dtype=None)
        return res


class ConditionalGAN(BaseModel):
    def name(self):
        return 'ConditionalGANModel'

    def __init__(self, opt):
        super(ConditionalGAN, self).__init__(opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_C = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        # Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
        use_parallel = False  # not opt.gan_type == 'wgan-gp'
        print("Use Parallel = ", "True" if use_parallel else "False")
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
            not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual
        )
        init_image = torch.zeros((1, 3, 256, 256), device='cuda')
        if self.isTrain:
            use_sigmoid = opt.gan_type == 'gan'
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel
            )
        if not self.isTrain or opt.continue_train:
            print("Load network.....")
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.lookhead_G = Lookahead(self.optimizer_G, k=5, alpha=0.5)
            self.lookhead_D = Lookahead(self.optimizer_D, k=5, alpha=0.5)

            self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1

            self.accumulation_steps = 8

            # define loss functions
            self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)
            self.edgeloss = local_max_gradients()  # EdgeLoss()
            self.soft_edge = soft_edge_loss()
            self.L1 = nn.L1Loss()

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputA = input['A' if AtoB else 'B']
        inputB = input['B' if AtoB else 'A']
        inputC = input['e']
        self.input_A.resize_(inputA.size()).copy_(inputA)
        self.input_C.resize_(inputC.size()).copy_(inputC)
        if isinstance(inputB, torch.Tensor):
            self.input_B.resize_(inputB.size()).copy_(inputB)
        elif isinstance(inputB, list):
            transform_tensor = transforms.ToTensor()
            inputB = transform_tensor(np.array(inputB))
            self.input_B.resize_(inputB.size()).copy_(inputB)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B, self.edge = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B)
        self.loss_D = self.loss_D  # /self.accumulation_steps
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(self.fake_B,
                                                        self.real_B) * self.opt.lambda_A  # self.contentLoss.get_loss

        self.edge_loss = self.contentLoss.get_loss(self.edge,
                                                   self.input_C) * self.opt.lambda_A  # self.opt.lambda_B #self.contentLoss.get_loss

        self.loss_G_soft_edge = 0.0  # self.edgeloss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A # self.opt.lambda_A#self.soft_edge(self.fake_B, self.real_B)

        self.loss_G = self.loss_G_GAN + self.loss_G_Content + self.edge_loss  # self.loss_G_soft_edge#+ self.edge_loss + self.loss_G_soft_edge
        self.loss_G = self.loss_G  # / self.accumulation_steps
        self.loss_G.backward()

    def optimize_parameters(self, i):
        self.forward()

        for iter_d in xrange(self.criticUpdates):
            self.optimizer_D.zero_grad()  # self.optimizer_D.zero_grad()
            # self.lookhead_D.zero_grad()
            self.backward_D()
            self.lookhead_D.step()  # self.optimizer_D.step()

        self.optimizer_G.zero_grad()  # self.optimizer_G.zero_grad()
        self.backward_G()
        self.lookhead_G.step()  # self.optimizer_G.step()

    # if ((i + 1) % self.accumulation_steps) == 0:

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_Content.item()),
                            ('D_real+fake', self.loss_D.item()),
                            ('G_edge', self.loss_G_soft_edge),
                            ('softloss', self.edge_loss)
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        # edge = util.tensor2im(self.edge.data)
        return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

    def get_tensor(self):
        return OrderedDict(
            [('Blurred_Train', self.real_A), ('Restored_Train', self.fake_B), ('Sharp_Train', self.real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        with open("learn_rate.txt", "a") as f:
            f.write("update learning rate: %f -> %f" % (self.old_lr, lr))
        self.old_lr = lr
