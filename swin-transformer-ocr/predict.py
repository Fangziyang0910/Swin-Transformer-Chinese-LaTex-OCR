import torch
import argparse
import time
from pathlib import Path

from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/test.yaml",
                        help="Experiment settings")
    parser.add_argument("--srcpath", "-sp", type=str, default="datasets_no_test/test/images/",
                        help="test/image路径")
    parser.add_argument("--ids","-i",type=str,default="datasets_no_test/test_ids.txt",help="test_ids文件路径")
    # parser.add_argument("--tokenizer", "-tk", type=str, required=True,
    #                     help="Load pre-built tokenizer")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="weights/checkpoints-epoch=279-val_overall_score=76.98470-accuracy=0.63202-val_bleu=0.79560-val_edit_distance=0.88192-val_loss=0.15343.ckpt",
                        help="Load model weight in checkpoint")
    parser.add_argument("--result","-r",type=str,default="dataset/result.txt",help="result.txt测试结果存放路径")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = cfg.height, cfg.width
    print(device)

    # load
    tokenizer = load_tokenizer(cfg.tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer).to(device)
    saved = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    collate = CustomCollate(cfg, tokenizer=tokenizer)

    test_image_dir = args.srcpath
    # if test_image_dir.is_dir():
    #     images = list(test_image_dir.glob("*.jpg")) + list(test_image_dir.glob("*.png"))
    # else:
    #     images = [test_image_dir]



    with open(args.ids, 'r', encoding='utf-8') as f:
        ids = f.read().split()

    images=[]
    for item in ids:
        images.append(test_image_dir+item+'.png')

    with open(args.result,'w',encoding='utf-8')as f:
        count=0
        for image in images:
            start = time.time()
            x = collate.ready_image(image, shape).to(device)
            f.write(" ".join(model.predict(x))+'\n')
            count+=1
            if(count%100==0):
                print(f'process {count}/{len(ids)}')

    #        print("[{}]sec | image_fn : {}".format(time.time() - start, model.predict(x)))
