import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import  util #html
from util.metrics import PSNR
from PIL import Image
import torchvision
import ssim_python
import numpy as np
import torch
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import datetime

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def ssim(img1, img2):
	img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
	img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
	# print(img1.shape)
	img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
	img2 = Variable(img2, requires_grad=False)
	ssim_value = ssim_python.ssim(img1, img2).item()
	return ssim_value

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
	if dim_order == 'hwc':
		y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
		cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
		cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
	else:
		y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
		cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
		cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
	return np.array([y, cb, cr]).transpose([1, 2, 0])

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataset_mode = "Gopro_test"
opt.model = "test"

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

avgPSNR = []
avgSSIM = []
counter = 0

for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break
	counter = i

	model.set_input(data)

	model.test()

	visuals = model.get_current_visuals()


	ten = model.get_tensor()
	print(ten['fake_B'].shape, " ", ten['real_B'].shape)
	print(torchPSNR(ten['fake_B'].detach().cpu(), ten['real_B'].detach().cpu()))
	p = compare_psnr(visuals['fake_B'], visuals['real_B'])

	avgPSNR.append(p)

	s = compare_ssim(visuals['fake_B'], visuals['real_B'], multichannel=True)

	avgSSIM.append(s)
	img_path = model.get_image_paths()
	print("#########################")
	print(i)
	print(p)
	print(s)
	print('process image... %s' % img_path)
	path = data["A_paths"][0].split('/')

	result = model.get_tensor()['fake_B']
	torchvision.utils.save_image(result[0].data, opt.results_dir)

psnr = np.mean(avgPSNR)
ssim = np.mean(avgSSIM)
print('PSNR = %f ssim = %f\n' % (psnr, ssim))
