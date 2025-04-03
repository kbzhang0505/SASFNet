

########### 简单重命名代码：把jpg格式转换成png格式 ##################
'''
import os

def convert_to_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            file_path = os.path.join(directory, filename)
            new_filename = os.path.splitext(filename)[0] + '.png'
            new_file_path = os.path.join(directory, new_filename)
            os.rename(file_path, new_file_path)

# 指定文件夹路径
directory = 'E:\\Desktop\\PetImages\\cat_dog_resized\\Cat'
convert_to_png(directory)
'''

################################ 重命名代码 #########################################################

'''
import os

# def print_list_dir(dir_path):
#     global j
#     dir_files=os.listdir(dir_path) #得到该文件夹下所有的文件
#     for file in  dir_files:
#         file = str(file) + '\\blur'
#         file_path=os.path.join(dir_path,file)  #路径拼接成绝对路径
#         if os.path.isdir(file_path):  #如果目录，就递归子目录
#             print_list_dir1(file_path)

def print_list_dir(dir_path):
    global j
    dir_files=os.listdir(dir_path) #得到该文件夹下所有的文件
    for file in  dir_files:  #这里的file已经是图片的名字了
        num = str(j).zfill(5)
        j = j + 1
        filename = "dog." + num  # 修改文件名的格式
        file_path=os.path.join(dir_path,file)  #路径拼接成绝对路径
        new_name=os.path.join(dir_path,filename+".jpg")
        os.rename(file_path, new_name)

        # if os.path.isfile(file_path): #如果是文件，就打印这个文件路径
        #     print(file_path)
        #     f.write(file_path + '\n')
            


if __name__ == '__main__':
    # f = open('F:\\scientific_research\\Python_Project\\multi_stage\\data\\test\\test_sharp.txt', 'w')
    # dir_path = 'F:\\scientific_research\\Python_Project\\GOPRO_Large\\test'
    dir_path = 'E:\\Desktop\\PetImages\\cat_dog_resized\\Dog'
    global j
    j = 0
    print_list_dir(dir_path)
'''

################################### 生成路径代码 #########################################################
'''
import os

def print_list_dir(dir_path):
    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
    for file in dir_files:
        file = str(file) + '/sharp'
        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
        if os.path.isdir(file_path):  # 如果目录，就递归子目录
            print_list_dir1(file_path)


def print_list_dir1(dir_path):
    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
    for file in dir_files:  # 这里的file已经是图片的名字了
        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
        
        if os.path.isfile(file_path): #如果是文件，就打印这个文件路径
            print(file_path)
            f.write(file_path + '\n')


if __name__ == '__main__':
    f = open('/opt/data/private/cj/Gopro/data/test/test_sharp.txt', 'w')
    dir_path = '/opt/data/private/cj/GoPro_MIMONet/test/sharp'

    print_list_dir(dir_path)
'''

'''
import os

def generate_image_paths(folder_path, output_file):
    # 获取文件夹下所有文件的路径
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

    # 过滤出图片文件并按照文件名排序
    image_paths = sorted(
        [file_path for file_path in file_paths if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif'))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    # image_paths = sorted([file_path for file_path in file_paths if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif'))], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # 将图片路径写入文本文件
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            f.write(image_path + '\n')

# 调用函数生成图片路径文件
folder_path = '/opt/data/private/cj/GoPro_MIMONet/test/blur'
output_file = '/opt/data/private/cj/GoPro_MIMONet/test/new_test_blur.txt'
generate_image_paths(folder_path, output_file)
'''

################################ 移动文件夹代码 ######################################################
'''
import os
import shutil
# def print_list_dir(dir_path, dest):
#     dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
#     for file in dir_files:
#         file = str(file) + '\\blur'
#         file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
#         if os.path.isdir(file_path):  # 如果目录，就递归子目录
#             print_list_dir1(file_path, dest)


def print_list_dir(dir_path, dest):
    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
    for file in dir_files:  # 这里的file已经是图片的名字了
        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
        # source_path = os.path.join(root, file)
        destination_path = os.path.join(dest, file)
        shutil.move(file_path, destination_path)
        # 判断停止移动条件
        filename = os.path.splitext(os.path.basename(file))[0]
        last_four_digits = filename[-5:]
        if last_four_digits > '10623':
            break


if __name__ == '__main__':
    # dir_path = "F:\\scientific_research\\Python_Project\\GOPRO_Large\\test"
    # dest = "F:\\scientific_research\\Python_Project\\GOPRO_Large\\data\\test\\blur"
    dir_path = "E:\\Desktop\\PetImages\\cat_dog_resized\\Dog"
    dest = "E:\\Desktop\\PetImages\\cat_dog_resized\\train"
    print_list_dir(dir_path, dest)
'''



############################ 生成txt文件代码，已检查无误！！！！！！##################################

import os
from natsort import natsorted

#定义一个函数，用于遍历所有命名为blur的文件夹，并将其中的图片路径依次保存在同一个txt文件中
def find_blur_dirs(root_path):
    blur_dirs = []
    for root, dirs, files in os.walk(root_path):
        for name in dirs:
            ############################################### 要改！！###################################################
            if name == "softedge":
                blur_dirs.append(os.path.join(root, name))

    if not blur_dirs:
        print("未找到命名为 sharp 的文件夹！")
        return

    sorted_blur_dirs = sorted(blur_dirs, key=lambda x: (
        os.path.basename(os.path.dirname(x)), os.path.basename(x)))
    print("找到 {} 个符合要求的 blur 文件夹：\n{}".format(len(sorted_blur_dirs), "\n".join(sorted_blur_dirs)))
    ############################################### 要改！！###################################################
    output_file = os.path.join(root_path, "gt_softedge_images.txt")
    with open(output_file, "w") as f:
        for blur_dir in sorted_blur_dirs:
            blur_dir_name = os.path.basename(os.path.dirname(blur_dir))
            for root, dirs, files in os.walk(blur_dir):
                sorted_files = natsorted(files)
                for file in sorted_files:
                    file_name, ext = os.path.splitext(file)
                    if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        ############################################### 要改！！###################################################
                        f.write("{}{}{}{}{}\n".format("/opt/data/private/cj/Gopro/data/train/", blur_dir_name,"/softedge/",  file_name, ext))
                        print("已将图片文件 {} 写入到 {}".format(file, output_file))

    print("所有图片的路径已按顺序保存到 {}".format(output_file))

#定义另外一个函数，从txt文件中按行读取文件，并返回视频文件名和对应的路径列表
def read_txt_file(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()
    f.close()

    file_paths = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        file_name = os.path.basename(line).split('.', 1)[0]
        if file_name in file_paths.keys():
            file_paths[file_name].append(line)
        else:
            file_paths[file_name] = [line]

    return file_paths

#使用示例
############################################### 要改！！###################################################
root_path = "/opt/data/private/cj/Gopro/data/train"
#遍历文件夹生成txt文件
find_blur_dirs(root_path)
# txt_file_path = os.path.join(root_path, "blur_images.txt")
#按行读取txt文件，并排序
# file_paths = read_txt_file(txt_file_path)
# for file_name, paths in sorted(file_paths.items()):
#     print(file_name)
#     sorted_paths = natsorted(paths)
#     for path in sorted_paths:
#         print("  ", path)

