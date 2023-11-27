import os
import shutil
import os
import cv2
import numpy as np
from PIL import Image
import argparse

#得到集合txt：train.txt,直接写在同一行
#
def copy_allfiles(src,dest):
#src:原文件夹；dest:目标文件夹
  src_files = os.listdir(src)
  for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)

def build_dataset(oldpath,newpath):
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    traintxt = newpath + '/train.txt'
    trainpath = oldpath+'/train/labels'
    files = os.listdir(trainpath)
    files.sort(key=lambda x: int(x[:-4]))
    for file in files:
        with open(trainpath + '/' + file, encoding='utf-8') as f:
            lines = f.readlines()
        data = ''
        for line in lines:
            data += line.rstrip()
        with open(traintxt, 'a', encoding='utf-8') as f:
            fileName = file[:-4] + '.png'
            f.write(fileName + '\t' + data + '\n')
    valpath = oldpath +'/dev/labels'
    valtxt = newpath  + '/val.txt'
    files = os.listdir(valpath)
    files.sort(key=lambda x: int(x[:-4]))
    for file in files:
        with open(valpath + '/' + file, encoding='utf-8') as f:
            lines = f.readlines()
        data = ''
        for line in lines:
            data += line.rstrip()
        with open(valtxt, 'a', encoding='utf-8') as f:
            fileName = file[:-4] + '.png'
            f.write(fileName + '\t' + data + '\n')
    if not os.path.isdir(newpath+'/images'):
        os.mkdir(newpath+'/images')
    oldpath1 = oldpath+'/train/images'
    oldpath2 = oldpath+'/dev/images'
    newpath = newpath+'/images'
    copy_allfiles(oldpath1, newpath)
    copy_allfiles(oldpath2, newpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oldpath', type=str, help='原始数据文件夹路径')
    parser.add_argument('--newpath', type=str, help='新数据文件夹路径')
    args = parser.parse_args()

    oldpath = args.oldpath
    newpath = args.newpath
    build_dataset(oldpath, newpath)

# if not os.path.isdir('./dataset'):
#     os.mkdir('./dataset')
# traintxt='./dataset/'+'train.txt'
# trainpath ='./datasets_no_test/train/labels'
# files= os.listdir(trainpath)
# files.sort(key=lambda x:int(x[:-4]))
#
# for file in files:
#     with open(trainpath+'/'+file, encoding='utf-8')as f:
#         lines = f.readlines()
#     data = ''
#     for line in lines:
#         data += line.rstrip()
#     with open(traintxt, 'a', encoding='utf-8') as f:
#         fileName = file[:-4] + '.png'
#         f.write(fileName + '\t' + data + '\n')
# #得到集合txt：val.txt,直接写在同一行
# valpath='./datasets_no_test/dev/labels'
# valtxt='./dataset/'+'val.txt'
# files= os.listdir(valpath)
# files.sort(key=lambda x:int(x[:-4]))
#
# for file in files:
#     with open(valpath+'/'+file, encoding='utf-8')as f:
#         lines = f.readlines()
#     data = ''
#     for line in lines:
#         data += line.rstrip()
#     with open(valtxt, 'a', encoding='utf-8') as f:
#         fileName = file[:-4] + '.png'
#         f.write(fileName + '\t' + data + '\n')

#将图片放在一起


if not os.path.isdir('./dataset/images'):
    os.mkdir('./dataset/images')
oldpath1='./datasets_no_test/train/images'
oldpath2='./datasets_no_test/dev/images'
newpath='./dataset/images'
copy_allfiles(oldpath1, newpath)
copy_allfiles(oldpath2, newpath)

# img = Image.open('./datasets_no_test/resized_train_match/resized_2.png.png')  # 打开图片
# im = np.array(img)  # 转化为ndarray对象
#
# im1 = np.concatenate((im, im), axis=0)  # 纵向拼接
# im2 = np.concatenate((im, im), axis=1)  # 横向拼接
#
# # 生成图片
# img1 = Image.fromarray(im1)
# img2 = Image.fromarray(im2)
#
# # 保存图片
# img1.save('test1.jpg')
#组合图片
# def combine_and_padding_images(folder_path):
#     file_list = os.listdir(folder_path)
#     parent_folder_path = os.path.dirname(os.path.dirname(folder_path))
#     im = np.array(file_list)
#     for i in len(file_list)-1:
#         # 构建完整的文件路径
#         file_name1,file_name2=file_list[i],file_list[i+1]
#         im1=cv2.imread(os.path.join(folder_path,file_name1))
#         im2=cv2.imread(os.path.join(folder_path,file_name2))
#         image = np.vstack((im1, im2))
#         resized_folder_path = os.path.join(parent_folder_path, 'resized_train_matched')
#         os.makedirs(resized_folder_path, exist_ok=True)
#         resized_file_path = os.path.join(resized_folder_path, f'resized_{file_name1+file_name2}')
#         cv2.imwrite(resized_file_path, image)
#     file_name1, file_name2 = file_list[len(file_list)], file_list[0]
#     im1 = cv2.imread(os.path.join(folder_path, file_name1))
#     im2 = cv2.imread(os.path.join(folder_path, file_name2))
#     image = np.vstack((im1, im2))
#     resized_folder_path = os.path.join(parent_folder_path, 'resized_train_matched')
#     os.makedirs(resized_folder_path, exist_ok=True)
#     resized_file_path = os.path.join(resized_folder_path, f'resized_{file_name1 + file_name2}')
#     cv2.imwrite(resized_file_path, image)
#
# script_dir = os.path.dirname(os.path.abspath(__file__))
#
# # 构建图片文件夹的相对路径
# folder_path = os.path.join(script_dir, 'datasets_no_test', 'resized_train_match')
# print('folder_path:',folder_path)
# combine_and_padding_images(folder_path)