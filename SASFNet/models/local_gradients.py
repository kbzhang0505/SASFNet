import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class local_max_gradients:
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

        self.weight_x = torch.nn.Parameter(data=self.kernel_x, requires_grad=False)
        self.weight_y = torch.nn.Parameter(data=self.kernel_y, requires_grad=False)

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



'''
#a = np.random.randn(720,1280)
i = torch.rand(1, 3, 30, 40)
h_x = i.size()[2]
w_x = i.size()[3]
r = F.pad(i, (0,1,0,0))[:, :, :, 1:]
l = F.pad(i, (1,0,0,0))[:,:,:,:w_x]
t = F.pad(i, (0,0,1,0))[:, :, :h_x, :]
b = F.pad(i, (0,0,0,1))[:,:,1:,:]
xgrad = torch.pow(torch.pow((r-l)*0.5,2) + torch.pow((t-b)*0.5,2), 0.5)
n = np.random.rand(1, 3, 720, 1280)
n = torch.Tensor(n)
print(n.size())
'''
'''

blur_img = torch.Tensor(blur_img)
sharp_img = torch.Tensor(sharp_img)

blur_img = blur_img.permute(2, 0, 1).unsqueeze(0)
sharp_img = sharp_img.permute(2, 0, 1).unsqueeze(0)

lmg = local_max_gradients()

g = lmg.get_loss(blur_img, sharp_img)
print(g)

mse = torch.nn.MSELoss()
print(mse(blur_img, sharp_img))
'''
