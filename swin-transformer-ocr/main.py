import os
import shutil
#这段代码是用来生成数学公式图片与其对应的编号的匹配文件。具体操作如下：
# 定义数据集路径和子集名称。
#
# 定义公式文件夹路径，如果没有匹配文件夹则创建一个。
#
# 遍历每个子集的公式文件，读取其中的每一行公式条目。
#
# 对于每个公式条目，生成一个对应的编号和文件名，并将其写入相应子集的匹配文件中。
#
# 最终生成的匹配文件格式为：每行一个数学公式图片的文件名和其对应的编号，中间用空格隔开。
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
        data += line.rstrip()
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
        data += line.rstrip()
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
