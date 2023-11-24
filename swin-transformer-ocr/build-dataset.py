import os
import shutil
# 根据数据集路径改data_path, 其他参数不需要改
# data_path = '210705'

#得到集合txt：train.txt,直接写在同一行
if not os.path.isdir('./dataset'):
    os.mkdir('./dataset')
traintxt='./dataset/'+'train.txt'
trainpath ='./datasets_no_test/train/labels'
files= os.listdir(trainpath)
files.sort(key=lambda x:int(x[:-4]))

for file in files:
    with open(trainpath+'/'+file, encoding='utf-8')as f:
        lines = f.readlines()
    data = ''
    for line in lines:
        data += line.rstrip().replace(' ', '')
    with open(traintxt, 'a', encoding='utf-8') as f:
        fileName = file[:-4] + '.png'
        f.write(fileName + '\t' + data + '\n')
#得到集合txt：val.txt,直接写在同一行
valpath='./datasets_no_test/dev/labels'
valtxt='./dataset/'+'val.txt'
files= os.listdir(valpath)
files.sort(key=lambda x:int(x[:-4]))

for file in files:
    with open(valpath+'/'+file, encoding='utf-8')as f:
        lines = f.readlines()
    data = ''
    for line in lines:
        data += line.rstrip().replace(' ', '')
    with open(valtxt, 'a', encoding='utf-8') as f:
        fileName = file[:-4] + '.png'
        f.write(fileName + '\t' + data + '\n')

#将图片放在一起
def copy_allfiles(src,dest):
#src:原文件夹；dest:目标文件夹
  src_files = os.listdir(src)
  for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)

if not os.path.isdir('./dataset/images'):
    os.mkdir('./dataset/images')
oldpath1='./datasets_no_test/train/images'
oldpath2='./datasets_no_test/dev/images'
newpath='./dataset/images'
copy_allfiles(oldpath1, newpath)
copy_allfiles(oldpath2, newpath)
