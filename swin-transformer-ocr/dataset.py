import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import albumentations as alb
from albumentations.pytorch import ToTensorV2



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, txt_fn):
        self.cfg = cfg
        self.images = []
        self.texts = []

        self.new_word_count = {}  # 新词出现次数的计数器


        # build one
        self.token_id_dict = {   # 文本到idx的映射  idx到文本的映射
            "token2id": {
                "[PAD]": cfg.pad_token,
                "[BOS]": cfg.bos_token,
                "[EOS]": cfg.eos_token,
                "[OOV]": cfg.oov_token
            },
            "id2token": {
                cfg.pad_token: "[PAD]",
                cfg.bos_token: "[BOS]",
                cfg.eos_token: "[EOS]",
                cfg.oov_token: "[OOV]"
            }
        }
        # 从txt中加载公式分词数据
        self.txt_dict = self.load_dictionary(self.cfg.txt_dict)

        skip_cnt, token_cnt = 0, 4 # 跳过计数和词计数
        # 直接先将t现有的xt分词加到字典里面
        for token in self.txt_dict:
            # 确保不重复添加
            if token not in self.token_id_dict['token2id']:
                self.token_id_dict['token2id'][token] = token_cnt
                self.token_id_dict['id2token'][token_cnt] = token
                token_cnt += 1

        with open(txt_fn, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    fn, text = line.strip().split('\t', 1) # 分成fn和文本
                except ValueError:
                    skip_cnt += 1
                    continue
                if cfg.max_seq_len < len(text) + 2:  # 超过最大句子长度的抛弃
                    # we will add [BOS] and [EOS]
                    skip_cnt += 1
                    continue
                self.images.append(fn)
                self.texts.append(text)
                if not cfg.load_tokenizer:
                    text_tokens = self.FMM_func(self.txt_dict, text)
                    for token in text_tokens:
                        if token not in self.token_id_dict["token2id"]:
                            self.new_word_count[token] = self.new_word_count.get(token, 0) + 1
                            if self.new_word_count[token] > 3:
                                self.token_id_dict["token2id"][token] = token_cnt
                                self.token_id_dict["id2token"][token_cnt] = token
                                token_cnt += 1

        print(f"{len(self.images)} data loaded. ({skip_cnt} data skipped)")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Read the image (use PIL Image to load unicode name images)
        if cfg.channels == 1, need to change alb transform methods
        """
        idx = idx % len(self.images)
        image = cv2.imread(str(Path(self.cfg.image_dir) / self.images[idx]))
        text = self.texts[idx]
        return image, text

    def FMM_func(self, user_dict, sentence):
        """
        正向最大匹配（FMM）
        :param user_dict: 词典
        :param sentence: 句子
        """
        # 词典中最长词长度
        max_len = max([len(item) for item in user_dict])
        start = 0
        token_list = []
        while start != len(sentence):
            index = start + max_len
            if index > len(sentence):
                index = len(sentence)
            for i in range(max_len):
                if (sentence[start:index] in user_dict) or (len(sentence[start:index]) == 1):
                    token_list.append(sentence[start:index])
                    # print(sentence[start:index], end='/')
                    start = index
                    break
                index += -1
        return token_list

    def load_dictionary(self, file_path):
        """
        从txt文件加载分词字典
        :param file_path: 分词文件的路径
        :return: 包含所有分词的集合
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用集合来存储分词，自动去除重复项
            word_set = set(word.strip() for word in file)
        return word_set
        
'''关于数据增强函数的注解：
alb.Resize:将输入图像调整为指定的尺寸大小（112x448）
alb.ShiftScaleRotate(shift_limit=0, scale_limit=(0., 0.15), rotate_limit=1,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7):随机进行平移、缩放和旋转变换。参数依次为平移的倍数，缩放的倍数，旋转的角度，边界的模式（填充边界用什么方法），插值方法（处理图像时，会有新的像素点），填充边界颜色，应用该函数的概率
alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5):理解为拉伸的操作，distort_limit为限制拉伸倍数
alb.GaussNoise(10, p=.2):增加高斯噪声，10为噪声强度，GaussNoise(var_limit=(10.0, 50.0), mean=None, always_apply=False, p=0.5)
alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2):亮度对比度调整，①亮度调整限制范围②对比度调整限制范围③是否根据图像最大值调整亮度
alb.ImageCompression(95, p=.3):图像压缩，压缩质量
alb.ToGray(always_apply=True):变灰图
alb.Normalize():归一化*
ToTensorV2():将图像从numpy数组格式转换为tensor格式'''

class CustomCollate(object):
    def __init__(self, cfg, tokenizer, is_train=True):
        self.cfg = cfg
        self.tokenizer = tokenizer
        #做图像的处理，resize，数据增强，转换为tensor，我不喜欢这个resize操作
        #将resized_image文件夹中的图像进行处理
        if is_train:
            self.transform = alb.Compose([
                        #alb.Resize(112, 448),
                        alb.PadIfNeeded(min_height=224, min_width=448,border_mode=cv2.BORDER_CONSTANT,value=[255, 255, 255]),
                        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(0., 0.15), rotate_limit=1,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        alb.GaussNoise(10, p=.2),
                        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        alb.ImageCompression(95, p=.3),
                        alb.ToGray(always_apply=True),
                        alb.Normalize(),
                        # alb.Sharpen()
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = alb.Compose(
                [
                    # alb.Resize(cfg.height, cfg.width),
                    alb.PadIfNeeded(min_height=224, min_width=448, border_mode=cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255]),
                    alb.ImageCompression(95, p=.3),
                    alb.ToGray(always_apply=True),
                    alb.Normalize(),
                    # alb.Sharpen()
                    ToTensorV2(),
                ]
            )

    def __call__(self, batch):  # 最终返回的是一个batch的图像和文本对
        """
        return:
            images, (seq, mask)
        """
        np_images, texts = zip(*batch)
        images = []
        for img in np_images:
            try:
                images.append(self.transform(image=img)["image"])
            except TypeError as e:
                continue
        images = torch.stack(images)
        labels = self.tokenizer.encode(texts)

        return (images, labels)

    def ready_image(self, image):
        if isinstance(image, Path):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError
        image = self.transform(image=image)["image"].unsqueeze(0)
        return image


class Tokenizer:
    def __init__(self, d):
        self.token2id = d["token2id"]
        self.id2token = d["id2token"]

    def __len__(self):
        return len(self.token2id)

    def FMM_func(self, user_dict, sentence):
        """
        正向最大匹配（FMM）
        :param user_dict: 词典，格式为 {词: 索引}
        :param sentence: 句子
        """
        max_len = max([len(word) for word in user_dict.keys()])
        start = 0
        token_list = []
        while start != len(sentence):
            index = start + max_len
            if index > len(sentence):
                index = len(sentence)
            for i in range(max_len):
                if (sentence[start:index] in user_dict) or (len(sentence[start:index]) == 1):
                    token_list.append(sentence[start:index])
                    start = index
                    break
                index -= 1
        return token_list

    def encode(self, texts: list):  #文本的encoder操作，处理成同样长度的序列，用mask标记空白部分
        """
        text:
            list of string form text
            [str, str, ...]
        return:
            tensors
        """
        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]
        oov = self.token2id["[OOV]"]

        ids = []
        for text in texts:
            text = self.FMM_func(self.token2id, text)  # 先进行分词，然后再转为emmbeding
            encoded = [bos,]
            for token in text:
                try:
                    encoded.append(self.token2id[token])
                except KeyError:
                    encoded.append(oov)
            encoded.append(eos)
            ids.append(torch.tensor(encoded))

        seq = pad_sequence(ids, batch_first=True, padding_value=pad)
        mask = torch.zeros_like(seq)
        for i, encoded in enumerate(ids):
            mask[i, :len(encoded)] = 1

        return seq.long(), mask.bool()

    def decode(self, labels): #解码部分
        """
        labels:
            [B, L] : B for batch size, L for Sequence Length
        """

        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]
        oov = self.token2id["[OOV]"]

        texts = []
        for label in labels.tolist():
            text = ""
            for id in label:
                if id == bos:
                    continue
                elif id == pad or id == eos:
                    break
                else:
                    text += self.id2token[id]

            texts.append(text)

        return texts
