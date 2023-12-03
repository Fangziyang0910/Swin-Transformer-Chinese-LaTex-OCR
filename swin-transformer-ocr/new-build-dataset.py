import shutil
import os
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
#假设给的数据集格式不变，模型所需数据集格式也不变
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
            # data += line.rstrip().replace(' ', '')
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
            # data += line.rstrip().replace(' ', '')
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
    parser.add_argument('--oldpath', type=str, default="origin_dataset1", help='原始数据文件夹路径')
    parser.add_argument('--newpath', type=str, default="dataset1", help='新数据文件夹路径')
    args = parser.parse_args()

    oldpath = args.oldpath
    newpath = args.newpath
    build_dataset(oldpath, newpath)