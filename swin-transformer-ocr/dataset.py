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

        skip_cnt, token_cnt = 0, 4 # 跳过计数和词计数
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
                if not cfg.load_tokenizer: # 这地方个要改掉，传进来的应该是一个分过词的结果，然按照后空格拆分，然后在这里构建成字典
                    for token in text:
                        if token not in self.token_id_dict["token2id"]:
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

class CustomCollate(object):
    def __init__(self, cfg, tokenizer, is_train=True):
        self.cfg = cfg
        self.tokenizer = tokenizer
        #做图像的处理，resize，数据增强，转换为tensor，我不喜欢这个resize操作
        if is_train:
            self.transform = alb.Compose([
                        alb.Resize(112, 448),
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
                    alb.Resize(cfg.height, cfg.width),
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
