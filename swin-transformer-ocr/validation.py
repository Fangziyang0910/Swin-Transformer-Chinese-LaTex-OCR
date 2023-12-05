import torch
import argparse
import time
from pathlib import Path

from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate

import nltk
import Levenshtein

import torch
import pytorch_lightning as pl
import argparse
from pathlib import Path

from models import SwinTransformerOCR
from dataset import CustomDataset, CustomCollate, Tokenizer
from utils import load_setting, save_tokenizer, CustomTensorBoardLogger, load_tokenizer

from torch.utils.data import DataLoader

from datetime import datetime
import sys
import logging
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/task_total.yaml",
                        help="Experiment settings")
    parser.add_argument("--tokenizer", "-tk", type=str, required=True,
                        help="Load pre-built tokenizer")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Load model weight in checkpoint")
    parser.add_argument("--load_tokenizer", "-bt", type=str, default="",
                        help="Load pre-built tokenizer")
    parser.add_argument("--num_workers", "-nw", type=int, default=16,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="Batch size for training and validate")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    device = 'cuda' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load
    tokenizer = load_tokenizer(cfg.tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer).to(device)
    saved = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    val_set = CustomDataset(cfg, cfg.val_data)
    collate = CustomCollate(cfg, tokenizer=tokenizer)

    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, 
                                  shuffle=True,
                                  collate_fn=collate)
    
    outputs=[]
    # print('test1')
    i=0
    n=len(valid_dataloader)
    val_losses=0.
    avg_bleu_scores=0.
    avg_edit_distances=0.
    accs=0.
    overalls=0.
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = "./logging/"+f"overall_{current_time}.txt"
    file=open(log_filename, 'w')
    sys.stdout = file
    
    for x,y in valid_dataloader:
        # if i==9:
        #     break
        print('batch {}'.format(i))
        i+=1
        x=x.to(device)
        tgt_seq, tgt_mask = y
        tgt_seq=tgt_seq.to(device)
        tgt_mask=tgt_mask.to(device)
        encoded = model.encoder(x)
        loss = model.decoder(tgt_seq, mask=tgt_mask, context=encoded)
        t1=time.time()
        dec = model.decoder.generate_v3((torch.ones(x.size(0),1)*model.bos_token).long().to(x.device), model.max_seq_len,
                                    eos_token=model.eos_token, context=encoded, temperature=model.temperature)
        t2=time.time()
        gt = model.tokenizer.decode(tgt_seq)
        # print(gt)
        pred = model.tokenizer.decode(dec)
        # print(pred)
        # break
        # assert len(gt) == len(pred)

        acc = sum([1 if gt[i] == pred[i] else 0 for i in range(len(gt))]) / x.size(0)
        bleu_score = sum([nltk.translate.bleu_score.sentence_bleu([gt[i]], pred[i]) for i in range(len(gt))]) / x.size(0)
        edit_distance = sum([(1. - (Levenshtein.distance(gt[i], pred[i]) / max(len(gt[i]), len(pred[i]), 1))) for i in range(len(gt))]) / x.size(0)

        # 计算综合指标
        avg_three_metric = (acc + bleu_score + edit_distance) * 100. / 3
        
        print(f'{t2-t1}sec  {i} {loss} {bleu_score} {edit_distance} {acc} {avg_three_metric}')

        # val_losses+=loss
        # avg_bleu_scores+=bleu_score
        # avg_edit_distances+=edit_distance
        # accs+=acc
        # overalls+=avg_three_metric

    # val_loss = val_losses/i #sum([x['val_loss'] for x in outputs]) / len(outputs)
    # acc = accs/i #sum([x['acc'] for x in outputs]) / len(outputs)

    # 由于现在指标是批次级别的平均值，直接计算整个验证集的平均值
    # avg_bleu_score = avg_bleu_scores/i #sum([x['avg_bleu_score'] for x in outputs]) / len(outputs)
    # avg_edit_distance = avg_edit_distances/i #sum([x['avg_edit_distance'] for x in outputs]) / len(outputs)

    # avg_three_metric = overalls/i #sum([x['overall'] for x in outputs]) / len(outputs)

    # wrong_cases = []
    # for output in outputs:
    #     for i in range(len(output['results']['gt'])):
    #         gt = output['results']['gt'][i]
    #         pred = output['results']['pred'][i]
    #         if gt != pred:
    #             wrong_cases.append("|gt:{}/pred:{}|".format(gt, pred))
    # wrong_cases = random.sample(wrong_cases, min(len(wrong_cases), model.cfg.batch_size//2))

    # result = {'val_loss': val_loss,
    #         'accuracy': acc,
    #         'val_bleu': avg_bleu_score,
    #         'val_edit_distance': avg_edit_distance,
    #         'val_overall_score': avg_three_metric}
    # result = validation_epoch_end(outputs)

    # print(result)
    # target = Path(cfg.target)
    # if target.is_dir():
    #     target = list(target.glob("*.jpg")) + list(target.glob("*.png"))
    # else:
    #     target = [target]

    # for image_fn in target:
    #     start = time.time()
    #     x = collate.ready_image(image_fn)
    #     print("[{}]sec | image_fn : {}".format(time.time()-start, model.predict(x)))
    sys.stdout = sys.__stdout__
    