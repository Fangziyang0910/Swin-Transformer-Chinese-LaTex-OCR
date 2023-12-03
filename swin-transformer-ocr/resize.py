import shutil
import os
import cv2
import argparse
#python resize.py --path dataset --target_ratio 2 --target_width 500
#folder_path:数据集图片存放路径
def resize_images(path, target_ratio,target_width):
    # 获取文件夹中的所有文件
    file_list = os.listdir(path+'/images')
    file_list.sort(key=lambda x: int(x[:-4]))
    parent_folder_path = path
    for file_name in file_list:
        # 构建完整的文件路径
        file_path = os.path.join(path+'/images',file_name)
            # 使用 OpenCV 读取图片
        image = cv2.imread(file_path)
            # 获取图片的长宽
        height, width, _ = image.shape

            # 计算长宽比
        ratio = width / height
        if ratio >= target_ratio:
            new_width=target_width
            new_height=int(new_width/ratio)
            resized_image=cv2.resize(image,(new_width,new_height))
            resized_folder_path = os.path.join(parent_folder_path, 'resized_images')
            os.makedirs(resized_folder_path, exist_ok=True)
            resized_file_path = os.path.join(resized_folder_path, f'{file_name}')
            cv2.imwrite(resized_file_path, resized_image)
        else:
            new_height=int(target_width/target_ratio)
            new_width=int(ratio*new_height)
            resized_image=cv2.resize(image,(new_width,new_height))
            resized_folder_path = os.path.join(parent_folder_path, 'resized_images')
            os.makedirs(resized_folder_path, exist_ok=True)
            resized_file_path = os.path.join(resized_folder_path, f'{file_name}')
            cv2.imwrite(resized_file_path, resized_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="dataset1", help='dataset路径')
    parser.add_argument('--target_ratio', type=float, default=2, help='目标宽高比')
    parser.add_argument('--target_width', type=int, default=448, help='目标宽长')
    args = parser.parse_args()
    path =args.path
    target_ratio=args.target_ratio
    target_width=args.target_width
    resize_images(path,target_ratio,target_width)
# dir_path = os.path.dirname(os.path.abspath(__file__))
# path='./dataset'
# target_ratio=2
# target_width=500
# resize_images(path,target_ratio,target_width)