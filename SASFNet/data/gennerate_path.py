import os

with open("train.txt", "r") as f:
	test_path = f.readlines()

for i in test_path:
	k = i.split()
	with open("train_1080_blur.txt", "a") as h:
		h.write(k[0]+"\n")
	with open("train_1080_sharp.txt", "a") as j:
		j.write(k[1] + "\n")

