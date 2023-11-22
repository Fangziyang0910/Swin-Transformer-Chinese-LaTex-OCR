import os
import shutil

# 这段代码的目的是从图像目录中找出与标签目录中相匹配的图像文件，
# 并将这些匹配的图像复制到指定的输出目录中。

label_dir = '../../images/formulas_labels_fmm/'
# label_dir = './data/math_210421/formula_labels_210421_no_chinese/'
image_dir = '../../images/output_images/'

output_dir ='../../images/formulas_images_fmm/'

label_name_list = os.listdir(label_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for i in range(len(label_name_list)):
    label_name_list[i] = label_name_list[i][:-4]

# print(label_list)

image_name_list = os.listdir(image_dir)

for image_name in image_name_list:
    if image_name[:-4] in label_name_list:
        print(image_name)
        shutil.copy(image_dir + image_name, output_dir + image_name)