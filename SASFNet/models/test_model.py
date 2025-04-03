import torch
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import datetime


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False, #
                                      opt.learn_residual)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.real_B = input['B']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)
        return self.real_A, self.fake_B


    # get image paths
    def get_image_paths(self):
        return self.image_paths[-1].split("/")[-1].rstrip()

    def get_tensor(self):
        return OrderedDict([('real_B', self.real_B), ('fake_B', self.fake_B[0])])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_B.data)
        fake_B = util.tensor2im(self.fake_B[0].data)
        return OrderedDict([('real_B', real_A), ('fake_B', fake_B)])
