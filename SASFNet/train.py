import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from multiprocessing import freeze_support # import pkg
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
# import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr



def train(opt, data_loader, model, visualizer):
	dataset = data_loader.load_data() # 加载数据集
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0
	b_p = 0.0
	b_s = 0.0
	p = 0.0
	s = 0.0
	count = 0
	mean_gan_error = 0.0
	mean_G_L1_error = 0.0
	mean_D_real_error = 0.0
	mean_G_edge_error = 0.0
	mean_G_softedge = 0.0
	# # 生成画布
	# plt.figure(figsize=(8, 6), dpi=80)
	#
	# # 打开交互模式
	# plt.ion()

	#x = []
	#to_loss = []
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time() # 记录时间
		epoch_iter = 0
		
		for i, data in enumerate(dataset):
			count = i
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			#print(type(data['B']))
			model.set_input(data)
			model.optimize_parameters(i)

			results = model.get_current_visuals()

			# psnrMetric = skimage.measure.compare_psnr(bi_lstm_results['Restored_Train'], bi_lstm_results['Sharp_Train'])
			psnrMetric = compare_psnr(results['Restored_Train'], results['Sharp_Train'])
			p += psnrMetric
			#ssimMetric = SSIM(bi_lstm_results['Restored_Train'], bi_lstm_results['Sharp_Train'])
			#print(bi_lstm_results['Restored_Train'].shape)
			ssimMetric = compare_ssim(results['Restored_Train'], results['Sharp_Train'], multichannel=True)
			# ssimMetric = skimage.measure.compare_ssim(bi_lstm_results['Restored_Train'], bi_lstm_results['Sharp_Train'], multichannel=True)
			s += ssimMetric

			if total_steps % opt.display_freq == 0:
				print('PSNR on Train = %f, SSIM = %f' % (psnrMetric, ssimMetric))
				#visualizer.display_current_results(bi_lstm_results, epoch)

			errors = model.get_current_errors()
			mean_gan_error += errors['G_GAN']
			mean_G_L1_error += errors['G_L1']
			mean_D_real_error += errors['D_real+fake']
			mean_G_edge_error += errors['G_edge']
			mean_G_softedge += errors['softloss']
			if total_steps % opt.print_freq == 0:
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				# model.save(total_steps)
				if opt.display_id > 0:
					pass
					#visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.save("latest")
		# print(count)
		mean_gan_error /= count
		mean_G_L1_error /= count
		mean_D_real_error /= count
		mean_G_edge_error /= count
		mean_G_softedge /= count
		p /= count
		s /= count
		print("#########################################################")
		print("epoch: %d  mean_gan_error: %f  mean_G_L1_error: %f  mean_D_real_error: %f mean_G_edge_error: %f softloss %f" % (epoch, mean_gan_error, mean_G_L1_error, mean_D_real_error, mean_G_edge_error, mean_G_softedge))
		print(" P : %f  S : %f \n" % (p, s))
		if s > b_s:
			model.save("train_best_ssim")
			b_s = s
		# with open("train_loss_branch_edge.txt", "a") as g:
		with open("train_loss_branch_edge_primary.txt", "a") as g:
			g.write("epoch: %d  mean_gan_error: %f  mean_G_L1_error: %f  mean_D_real_error: %f mean_G_edge_error: %f softloss %f P : %f  S : %f \n" % (epoch, mean_gan_error, mean_G_L1_error, mean_D_real_error, mean_G_edge_error, mean_G_softedge, p, s))
		mean_gan_error = 0
		mean_G_L1_error = 0
		mean_D_real_error = 0
		mean_G_edge_error = 0
		mean_G_softedge = 0
		count = 0
		p = 0
		s = 0
		if epoch % opt.save_epoch_freq == 0:# % opt.save_epoch_freq
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save('latest')
			model.save(epoch)
			
			#print("############## Test... ###############")
			#f = os.popen("./shell.sh")

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()
	# # 关闭交互模式
	# plt.ioff()
	#
	# # 图形显示
	# plt.show()


if __name__ == '__main__':
	freeze_support() # Run code for process object if this in not the main process

	# python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)

	opt = TrainOptions().parse() # get paras
	# opt.dataroot = 'D:\Photos\TrainingData\BlurredSharp\combined' # datasource root
	opt.learn_residual = True
	opt.resize_or_crop = "crop"
	opt.fineSize = 256
	opt.gan_type = "gan" # learning set; wgan
	# opt.which_model_netG = "unet_256"
	opt.continue_train = True
	# default = 5000
	opt.save_latest_freq = 100 # save freq

	# default = 100
	opt.print_freq = 50
	# pytorch tensorboard
	# tb_writer = SummaryWriter(log_dir="softedge")
	data_loader = CreateDataLoader(opt) # load data
	model = create_model(opt)
	visualizer = Visualizer(opt)
	train(opt, data_loader, model, visualizer)
