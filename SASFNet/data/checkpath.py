import os

'''
with open("train_blur.txt", "r") as f:
	train_blur = f.readlines()

with open("train_sharp.txt", "r") as h:
	train_sharp = h.readlines()
	
with open("test_blur.txt", "r") as j:
	test_blur = j.readlines()

with open("test_sharp.txt", "r") as k:
	test_sharp = k.readlines()
'''
with open("RealBlur_J_test_list_blur.txt", "r") as k:
	edge_sharp = k.readlines()

for i in range(len(edge_sharp)):
	with open("test_realblur_blur.txt", "a") as l:
		l.write("/media/ubuntu/PortableSSD" + edge_sharp[i][17:])


with open("RealBlur_J_test_list_sharp.txt", "r") as k:
	real_sharp = k.readlines()

for i in range(len(edge_sharp)):
	with open("test_realblur_sharp.txt", "a") as l:
		l.write("/media/ubuntu/PortableSSD" + real_sharp[i][17:])

'''
	with open("train_edge_1080.txt", "a") as l:
		l.write("/media/ubuntu/PortableSSD/Gopro/data" + test_blur[i][35:71] + " " + "/media/ubuntu/PortableSSD/Gopro/data" + test_sharp[i][35:])

for i in range(len(train_blur)):
	with open("train.txt", "a") as p:
		p.write("/media/ubuntu/PortableSSD/Gopro/data" + train_blur[i][35:72] + " " + "/media/ubuntu/PortableSSD/Gopro/data" + train_sharp[i][35:])
'''
