import torch
import random
import pytorch_lightning as pl

from x_transformers import *
from x_transformers.autoregressive_wrapper import *

from timm.models.swin_transformer import SwinTransformer

import utils

import nltk
import Levenshtein

import numpy as np


class SwinTransformerOCR(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.encoder = CustomSwinTransformer( img_size=(cfg.height, cfg.width),
                                        patch_size=cfg.patch_size,
                                        in_chans=cfg.channels,
                                        num_classes=0,
                                        window_size=cfg.window_size,
                                        embed_dim=cfg.encoder_dim,
                                        depths=cfg.encoder_depth,
                                        num_heads=cfg.encoder_heads
                                        )
        self.decoder = CustomARWrapper(
                        TransformerWrapper(
                            num_tokens=len(tokenizer),
                            max_seq_len=cfg.max_seq_len,
                            attn_layers=Decoder(
                                dim=cfg.decoder_dim,
                                depth=cfg.decoder_depth,
                                heads=cfg.decoder_heads,
                                **cfg.decoder_cfg
                            )),
                        pad_value=cfg.pad_token
                    )
        self.bos_token = cfg.bos_token
        self.eos_token = cfg.eos_token
        self.max_seq_len = cfg.max_seq_len
        self.temperature = cfg.temperature

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer)
        optimizer = optimizer(self.parameters(), lr=float(self.cfg.lr))

        if not self.cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
            scheduler = {
                'scheduler': scheduler, 'interval': "epoch", "name": "learning rate"
            }
            return [optimizer], [scheduler]
        elif hasattr(torch.optim.lr_scheduler, self.cfg.scheduler):
            scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler)
        elif hasattr(utils, self.cfg.scheduler):
            scheduler = getattr(utils, self.cfg.scheduler)
        else:
            raise ModuleNotFoundError

        scheduler = {
            'scheduler': scheduler(optimizer, **self.cfg.scheduler_param),
            'interval': self.cfg.scheduler_interval,
            'name': "learning rate"
            }
        return [optimizer], [scheduler]

    def forward(self, x):
        '''
        x: (B, C, W, H)
        labels: (B, S)

        # B : batch size
        # W : image width
        # H : image height
        # S : source sequence length
        # E : hidden size
        # V : vocab size
        '''

        encoded = self.encoder(x)
        dec = self.decoder.generate(torch.LongTensor([self.bos_token]*len(x))[:, None].to(x.device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec

    def training_step(self, batch, batch_num):
        x, y = batch
        tgt_seq, tgt_mask = y
        encoded = self.encoder(x)
        loss = self.decoder(tgt_seq, mask=tgt_mask, context=encoded)
        self.log("train_loss", loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_num):
        x, y = batch
        tgt_seq, tgt_mask = y
        encoded = self.encoder(x)
        loss = self.decoder(tgt_seq, mask=tgt_mask, context=encoded)
        dec = self.decoder.generate((torch.ones(x.size(0),1)*self.bos_token).long().to(x.device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        gt = self.tokenizer.decode(tgt_seq)
        pred = self.tokenizer.decode(dec)

        assert len(gt) == len(pred)

        acc = sum([1 if gt[i] == pred[i] else 0 for i in range(len(gt))]) / x.size(0)
        bleu_score = sum([nltk.translate.bleu_score.sentence_bleu([gt[i]], pred[i]) for i in range(len(gt))]) / x.size(0)
        edit_distance = sum([(1. - (Levenshtein.distance(gt[i], pred[i]) / max(len(gt[i]), len(pred[i]), 1))) for i in range(len(gt))]) / x.size(0)
        #
        # # 对批次中的每个样本进行计算
        # for i in range(x.size(0)):
        #
        #     # 计算 BLEU Score
        #     # 注意：BLEU Score 通常需要一个列表作为参考句子
        #     bleu_score = nltk.translate.bleu_score.sentence_bleu([gt[i]], pred[i])
        #     total_bleu_score += bleu_score
        #
        #     # 计算 Edit Distance
        #     edit_distance = Levenshtein.distance(gt[i], pred[i])
        #
        #     edit_distance = 1. - (edit_distance / max(len(gt[i]),len(pred[i]), 1))
        #
        #
        # # 计算平均指标
        # avg_bleu_score = total_bleu_score / x.size(0)
        # avg_edit_distance = total_edit_distance / x.size(0)
        #
        # avg_three_metric = (acc+avg_bleu_score+avg_edit_distance)*100./3

        # self.log('val_loss_step', loss)
        # self.log('accuracy_step', acc)
        # self.log('val_bleu_step', avg_bleu_score)
        # self.log('val_edit_distance_step', avg_edit_distance)
        # self.log('val_overall_step', avg_three_metric)

        def calculate_metrics(gt, pred):
            # 计算 BLEU Scores 和 Edit Distances
            bleu_scores = [nltk.translate.bleu_score.sentence_bleu([g], p) for g, p in zip(gt, pred)]
            edit_distances = [1. - (Levenshtein.distance(g, p) / max(len(g), len(p), 1)) for g, p in zip(gt, pred)]

            return bleu_scores, edit_distances

        # 示例数据
        # gt = ... # 实际标签的列表
        # pred = ... # 预测标签的列表
        # acc = ... # 准确率

        # 计算 BLEU Scores 和 Edit Distances
        bleu_scores, edit_distances = calculate_metrics(gt, pred)

        # 将列表转换为 NumPy 数组以利用向量化操作
        bleu_scores = np.array(bleu_scores)
        edit_distances = np.array(edit_distances)

        # 计算平均指标
        bleu_score = np.mean(bleu_scores)
        edit_distance = np.mean(edit_distances)

        # 计算综合指标
        avg_three_metric = (acc + bleu_score + edit_distance) * 100. / 3

        return {'val_loss': loss,
                'avg_bleu_score': bleu_score,
                'avg_edit_distance': edit_distance,
                'results' : {
                    'gt' : gt,
                    'pred' : pred
                    },
                'acc': acc,
                'overall': avg_three_metric
                }

    def validation_epoch_end(self, outputs):
        val_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        acc = sum([x['acc'] for x in outputs]) / len(outputs)

        # 由于现在指标是批次级别的平均值，直接计算整个验证集的平均值
        avg_bleu_score = sum([x['avg_bleu_score'] for x in outputs]) / len(outputs)
        avg_edit_distance = sum([x['avg_edit_distance'] for x in outputs]) / len(outputs)
        
        avg_three_metric = sum([x['overall'] for x in outputs]) / len(outputs)

        wrong_cases = []
        for output in outputs:
            for i in range(len(output['results']['gt'])):
                gt = output['results']['gt'][i]
                pred = output['results']['pred'][i]
                if gt != pred:
                    wrong_cases.append("|gt:{}/pred:{}|".format(gt, pred))
        wrong_cases = random.sample(wrong_cases, min(len(wrong_cases), self.cfg.batch_size//2))

        self.log('val_loss', val_loss)
        self.log('accuracy', acc)
        self.log('val_bleu', avg_bleu_score)
        self.log('val_edit_distance', avg_edit_distance)
        self.log('val_overall_score', avg_three_metric)

        # custom text logging
        self.logger.log_text("wrong_case", "___".join(wrong_cases), self.global_step)

    @torch.no_grad()
    def predict(self, image):
        dec = self(image)
        pred = self.tokenizer.decode(dec)
        return pred


class CustomSwinTransformer(SwinTransformer):
    def __init__(self, img_size=224, *cfg, **kwcfg):
        super(CustomSwinTransformer, self).__init__(img_size=img_size, *cfg, **kwcfg)
        self.height, self.width = img_size

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *cfg, **kwcfg):
        super(CustomARWrapper, self).__init__(*cfg, **kwcfg)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwcfg):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwcfg.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwcfg)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                # 对给定的 logits（模型的原始输出）进行 top-k 操作。
                # 具体而言，它将 logits 中除了前 k 个最大值之外的所有值设置为负无穷（`float('-inf')'）。
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                # entmax_bisect 是一个用于计算 Entmax 操作（带有可调整参数的 Softmax）的库中的函数。
                # Entmax 是 Softmax 的一种变体，允许用户通过参数调整输出的稀疏性。
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            # 从多项分布中采样。在这里，probs 张量表示一个多项分布的概率分布。
            # 从 probs 中进行一次多项式采样，返回的 sample 是包含采样结果的张量。
            # 1 是参数 num_samples，表示要采样的样本数量，这里是采样一个样本
            # sample 中的元素是被选中的类别的索引，这样就可以根据这个索引获取相应类别的信息
            sample = torch.multinomial(probs, 1)

            # 讲sample附加在out上，out参与下一个token预测
            out = torch.cat((out, sample), dim=-1)
            # mask: 输入的二进制掩码。
            # (0, 1): 表示填充的配置，其中 (0, 1) 意味着在最后一个维度的右侧填充一个元素，而在其他维度不进行填充。
            # value=True: 表示用 True 填充。
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out
    
    @torch.no_grad()
    def generate_val(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwcfg):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        
        topK=1

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwcfg.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        out=[out]
        
        for i in range(seq_len):
            
            out2=[]
            mask = mask[ :, -self.max_seq_len:]
            for j in range(len(out)):
                
                # eos_token is not None: 这是一个条件，检查是否提供了结束标记。如果 eos_token 不为 None，则说明需要检查结束标记。
                # (torch.cumsum(out == eos_token, 1)[:, -1] >= 1): 这部分代码使用了 PyTorch 的 torch.cumsum 函数，
                # 它计算了在 out == eos_token 中每个位置上的累积和。
                # 然后，[:, -1] 选择了每个序列中最后一个位置的值。最后，>= 1 检查每个序列是否至少包含一个结束标记。
                # .all(): 这是一个逻辑运算，检查整个张量是否都为 True。如果所有的序列都至少包含一个结束标记，那么这个条件为 True。
                # 如果以上条件为 True，则说明生成的序列中包含了结束标记，那么 break 语句将退出当前循环
                if eos_token is not None and (torch.cumsum(out[j] == eos_token, 1)[:, -1] >= 1).all():
                    out2.append(out[j])
                    continue
                
                x = out[j][ :, -self.max_seq_len:]
                

                logits = self.net(x, mask=mask, **kwcfg)[:, -1, :]

                if filter_logits_fn in {top_k, top_p}:
                    # 对给定的 logits（模型的原始输出）进行 top-k 操作。
                    # 具体而言，它将 logits 中除了前 k 个最大值之外的所有值设置为负无穷（`float('-inf')'）。
                    filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                elif filter_logits_fn is entmax:
                    # entmax_bisect 是一个用于计算 Entmax 操作（带有可调整参数的 Softmax）的库中的函数。
                    # Entmax 是 Softmax 的一种变体，允许用户通过参数调整输出的稀疏性。
                    probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

                # 从多项分布中采样。在这里，probs 张量表示一个多项分布的概率分布。
                # 从 probs 中进行一次多项式采样，返回的 sample 是包含采样结果的张量。
                # 1 是参数 num_samples，表示要采样的样本数量，这里是采样一个样本
                # sample 中的元素是被选中的类别的索引，这样就可以根据这个索引获取相应类别的信息
                sample = torch.multinomial(probs, 2)
                # print(out[j].shape)
                # print(sample.shape)
                # sample = torch.topk(probs,topK).indices

                # 讲sample附加在out上，out参与下一个token预测
                # out = torch.cat((out, sample), dim=-1)
                for ii in range(topK):
                    out2.append(torch.cat((out[j], sample[0,ii].reshape(1,-1)), dim=-1))     
                    print(out2)   
                
            # mask: 输入的二进制掩码。
            # (0, 1): 表示填充的配置，其中 (0, 1) 意味着在最后一个维度的右侧填充一个元素，而在其他维度不进行填充。
            # value=True: 表示用 True 填充。
            mask = F.pad(mask, (0, 1), value=True)
                
            out=out2

        out2 = []
        for i in range(len(out)):
            out2.append(out[i][:, t:])
        out = out2

        if num_dims == 1:
            out[0] = out[0].squeeze(0)

        self.net.train(was_training)
        return out[0]
