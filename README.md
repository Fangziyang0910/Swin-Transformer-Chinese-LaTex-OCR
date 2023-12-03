本项目基于Swin-Transformer实现中文和latex公式的混合识别。

## push注意事项

- 请不要上传数据集，包括原始数据集和处理过的数据集。请在gitignore中添加补充好数据集的路径
- 每次push之前请先pull，避免产生冲突
- 每次做了代码工作最好在这个readme中说明清楚工作结果。

数据集存放在swin-tranformer-ocr文件夹中，处理好的数据在该目录下的dataset文件夹中。请注意代码的路径尽量使用相对路径，并说清楚工作目录。

## 代码组织结构

```bash
./swin-transformer-ocr/
├─ checkpoint/   存放的是训练生成的模型权重文件（较大，不在git中）
├─ data-preprocess 这里是助教给的数据处理参考代码
│  ├─ ...
├─ dataset/   存放处理好的数据集
│  ├─ images
│  ├─ train.txt
│  ├─ val.txt
│  ├─ vocab.pkl
│  ├─ vocab.txt
├─ settings  存放的是模型配置文件
│  ├─ test.yaml
├─ run.py
└─ ...
```

## 使用方法

### 环境配置

1.先创建ssh连接密钥，然后克隆仓库。

```
git clone git@github.com:Fangziyang0910/Swin-Transformer-Chinese-LaTex-OCR.git
```

2.创建一个conda环境

```
conda create -n latex_ocr python=3.8
conda activate latex_ocr
```

3.进入并安装pytorch以及对应的依赖,pytorch请安装自己对应的pytorch版本

pytorch网址：https://pytorch.org/get-started/previous-versions/
安装pytorch,这个版本请改成自己的，并在安装后检查版本

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

4.安装依赖

```
cd Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr
pip install -r requirments.txt
```

### 放入数据集



1.将数据集自己放进来，放在swin..这个目录下,然后运行一次built-dataset.py(新的数据集进来要改路径)。有点慢请耐心等待，处理好的数据将会放在dataset文件夹中。

```
python new-build-dataset.py --oldpath datasets_no_test --newpath dataset
```
2.运行resize.py。在dataset中得到resized_images文件夹，用该文件夹作为图像文件夹参数传入

```
python resize.py --path dataset --target_ratio 2 --target_width 448
```

3.可以开始训练，

```
python run.py --batch_size==16 # 根据自己的gpu实力自己调
```

需要注意的是要在run代码的最后修改一下

这里的gpus将我注视掉的取消，把下面一行注视掉，这是因为我多卡训练的时候做的特殊处理，你们要用回默认配置。

```python
trainer = pl.Trainer(# gpus=device_cnt,   
                     gpus=[1,2],
                     max_epochs=cfg.epochs,
                     logger=logger,
                     num_sanity_val_steps=1,
                     strategy=strategy,
                     callbacks=[ckpt_callback, lr_callback],
                     resume_from_checkpoint=cfg.resume_train if cfg.resume_train else None)
```

## 文件说明

#### new_build_dataset.py

用于得到train.txt：将原本存储在上万个文件中的labels合并成一个文件

使用方法

```
python new-build-dataset.py --oldpath <oldpath> --newpath <newpath>
```

oldpath指向datasets_no_test

#### dataset.py

##### CustomDataset

用于导入vocab.txt以建立token表，vocab.txt的存储路径于settings/test.yaml中的txt_dict, 需要根据自己的存放位置进行修改test.yaml

需要修改 CustomCollate中的

```
alb.PadIfNeeded(min_height=224, min_width=448,border_mode=cv2.BORDER_CONSTANT,value=[255, 255, 255]),
```

将min_height和min_width与test.yaml中的height,width调整相同。

test.yaml中的image_dir需修改为使用resize.py处理后的（调整尺寸后的）数据集，路径如: "dataset/resized_images"

##### CustomCollate

用于对一个批次的样本数据进行数据增强，其调用返回的是一个批次的(images,labels)，images/labels是一个列表

##### Tokenizer

包含了一些方法用于文本处理和编码解码操作

#### get_useful_vocab.py

用于补充助教给的词表，补充结果已经写在vocab_useful.txt

#### predict.py

`python predict.py --srcpath <image文件夹路径> --ids <test_ids.txt路径> --checkpoint <权重文件路径> --result  <输出结果地址>`

## ToDoList

### 文本处理部分

#### 构建完整词库

词库的选择：助教给的latex公式字符库，里面包含了latex公式中的常见字符，字母和数字。
由于我们的数据是初高中试卷，应该用不到很多的特殊字符，所以初期暂时先用这个作为词库。

1. - [x] 我已经在dataset.py的CustomDataset类中添加了读取txt格式的词库的函数， 并用助教给的FMM替换了原始的分词方法，实现了基于给定txt词库的分词。
2. - [x] 计划将中文词库和公式词库合并在一起作为第二个任务的总词库，所以需要根据2中的基础分词构建新的完整的词库，也就是将中文加入进来。已经实现
3. - [x] 在构建新词库的代码中增加过滤器，**只有出现次数大于10词的新单词才加入词库。**
4. - [x] 将处理好的字典保存成pkl格式的词库并保存（代码自动完成）



11.24更新

1. - [x] 构建完整的大的公式词表（从别的latexocr中获取他们的dict，用我们代码中的util中的save——tokenizer来保存成自己的字典。也可以直接搜latex的官方帮助文档，手动添加一些符号）

     frequency.txt存储了大公式词表各单词使用频率

2. - [x] 去除掉ground true文本中所有的空格（已经完成）

3. - [x] 改写built-dataset函数，希望数据集路径成为一个单独的超参。(py文件为：new-build-dataset.py;参考命令：python new-build-dataset.py --oldpath datasets_no_test --newpath dataset)


### 图像处理部分

1. - [x] 统计的图像数据集图像尺寸分布情况
2. - [x]  对图像按比例进行resize，目标是尽量放大,将resize后的图片放到resized_images文件夹中
3. - [x]  对图像进行padding，padding成一个2：1长方形，padding函数在alb.Compose中与其他数据增强函数放在一起
4. - [x]  数据增强

### 模型部分

### 模型指标部分

1. - [ ] 生成文本质量指标包括bleu_score, edit_distance, token_accuracy，实现方法将参照项目 [LatexOCR](https://github.com/lukas-blecher/LaTeX-OCR)
2. - [ ] 指标计算部分集成在SwinTransformerOCR中
