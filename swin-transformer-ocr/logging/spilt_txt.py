file_path1 = "overall_2023-12-04_19-23-28.txt"  # 请替换成你的文本文件路径
file_path2 = "overall_2023-12-04_20-33-40.txt"  # 请替换成你的文本文件路径

# 打开文本文件
with open(file_path1, 'r') as file:
    overall=[]
    bleu_score=[] 
    edit_distance=[] 
    acc=[]
    # 逐行读取文件
    for line in file:
        # 使用 split 方法按空格划分每一行
        elements = line.split()

        # 提取最后一个元素
        if elements:
            overall.append(float(elements[-1]))
            bleu_score.append(float(elements[-4]))
            edit_distance.append(float(elements[-3]))
            acc.append(float(elements[-2]))
    print("without beam search", sum(overall)/len(overall))
    print(f'{sum(bleu_score)/len(bleu_score)}  {sum(edit_distance)/len(edit_distance)}  {sum(acc)/len(acc)}')
            
# 打开文本文件
with open(file_path2, 'r') as file:
    overall=[]
    bleu_score=[] 
    edit_distance=[] 
    acc=[]
    # 逐行读取文件
    for line in file:
        # 使用 split 方法按空格划分每一行
        elements = line.split()

        # 提取最后一个元素
        if elements:
            overall.append(float(elements[-1]))
            bleu_score.append(float(elements[-4]))
            edit_distance.append(float(elements[-3]))
            acc.append(float(elements[-2]))
    print("with beam search", sum(overall)/len(overall))
    print(f'{sum(bleu_score)/len(bleu_score)}  {sum(edit_distance)/len(edit_distance)}  {sum(acc)/len(acc)}')
