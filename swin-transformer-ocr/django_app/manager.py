# import PIL
# import os
# import numpy as np
import time
import threading
# from scipy.misc import imread
# from PIL import Image

# from model.img2seq import Img2SeqModel
# from model.utils.general import Config, run
# from model.utils.text import Vocab
# from model.utils.image import greyscale, crop_image, pad_image, downsample_image, TIMEOUT
# from model.utils.visualize_attention import clear_global_attention_slice_stack
# from model.utils.visualize_attention import readImageAndShape, vis_attention_slices, getWH
# from model.utils.visualize_attention import vis_attention_gif

import torch
# import argparse
import time
# from pathlib import Path

from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate


def getModelForPrediction():
    # # restore config and model
    # dir_output = "./results/full/"
    # config_vocab = Config(dir_output + "vocab.json")
    # config_model = Config(dir_output + "model.json")
    # vocab = Vocab(config_vocab)

    # model = Img2SeqModel(config_model, dir_output, vocab)
    # model.build_pred()
    # # model.restore_session(dir_output + "model_weights/model.cpkt")
    
    print('into getModelForPrediction')
    cfg = load_setting('/root/autodl-tmp/Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr/settings/task_total.yaml')
    print('cfg load success')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load
    tokenizer = load_tokenizer('/root/autodl-tmp/Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr/'+cfg.tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer).to(device)
    print('model init success')

    checkpoint='/root/autodl-tmp/Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr/checkpoints/checkpoints-epoch=96-val_overall_score=90.03386-accuracy=0.85575-val_bleu=0.88021-val_edit_distance=0.96506-val_loss=0.04882.ckpt'
    saved = torch.load(checkpoint, map_location=device)
    print('save load success')
    model.load_state_dict(saved['state_dict'])
    print('model load success')
    collate = CustomCollate(cfg, tokenizer=tokenizer)
    
    return model,collate


def predict_png(model,collate, png_path):
    cfg = load_setting('/root/autodl-tmp/Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr/settings/task_total.yaml')
    shape = cfg.height, cfg.width
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = collate.ready_image(png_path, shape).to(device)
    hyps = model.predict(img)[0]+'\n'

    # model.logger.info(hyps[0])

    return hyps


class ModelManager(object):
    """ModelManager is a warpper of Model. It extends model and provides more powerful methods."""
    _instance_lock = threading.Lock()  # 线程锁

    def __init__(self, model=None, collate=None):
        print("init ModelManager")
        if model is None:
            self.model,self.collate = getModelForPrediction()
        else:
            self.model,self.collate = model,collate
        print("init model")
        self.cfg=load_setting('/root/autodl-tmp/Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr/settings/task_total.yaml')

    @classmethod
    def instance(cls, *args, **kwargs):
        """多线程安全的单例模式"""
        print("instance")
        if not hasattr(ModelManager, "_instance"):
            with ModelManager._instance_lock:  # 为了保证线程安全在内部加锁
                if not hasattr(ModelManager, "_instance"):
                    ModelManager._instance = ModelManager(*args, **kwargs)
        return ModelManager._instance

    def predict_png(self, png_path):
        """
          Args:
            png_path(string): path to png
          Return:
            latex(string): predicted latex from png

        """
        print("predict_png")
        
        start=time.time()
        shape = self.cfg.height, self.cfg.width
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = self.collate.ready_image(png_path, shape).to(device)
        hyps = self.model.predict(img)[0]+'\n'
        end=time.time()
        
        print("finish prediction in {}".format(end-start))

        return hyps

    # def vis_png(self, png_path):
    #     dir_output = "./results/full/"
    #     img_path = png_path
    #     img, img_w, img_h = readImageAndShape(img_path)
    #     att_w, att_h = getWH(img_w, img_h)
    #     print("image path: {0} shape: {1}".format(img_path, (img_w, img_h)))
    #     clear_global_attention_slice_stack()
    #     hyps = self.model.predict(img)
    #     # hyps 是个列表，元素类型是 str, 元素个数等于 beam_search 的 bean_size
    #     # bean_size 在 `./configs/model.json` 里配置，预训练模型里取 2
    #     print(hyps[0])

    #     path_to_save_attention = dir_output+"vis/vis_"+img_path.split('/')[-1][:-4]
    #     vis_attention_slices(img_path, path_to_save_attention)
    #     gif_path = vis_attention_gif(img_path, path_to_save_attention, hyps)
    #     print(gif_path)
    #     return (hyps[0], gif_path)

    def statistic(self, method):
        print("start")
        print("end")

class DataManager:
    """DataManager is a warpper of data_generator. It provides default implementation."""

    def __init__(self, data_generator):
        super(DataManager, self).__init__()
        self.arg = arg


class Manager(ModelManager):
    """ModelManager is a warpper of Model. It provides default implementation."""

    def __init__(self, data_generator, model):
        super(Manager, self).__init__(model)
        self.data_generator = data_generator
