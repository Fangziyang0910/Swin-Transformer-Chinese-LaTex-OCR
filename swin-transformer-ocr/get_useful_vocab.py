import os


def FMM_func(user_dict, sentence):
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


input_label_dir = 'data/origin_dataset2/train/labels/'

label_name_list = os.listdir(input_label_dir)
with open('data/vocab_new.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().split()

counter = {word: 0 for word in vocab}

count = 0
file_count = 0
for label in label_name_list:
    label_file_name = input_label_dir + label
    with open(label_file_name, 'r', encoding='utf-8') as f1:
        content = f1.read()
        # 除去token_list中的空格和空元素
        token_list = [token for token in FMM_func(vocab, content) if token not in ['', ' ']]
        count += len(token_list)
        for token in token_list:
            if token in vocab:
                counter[token] += 1
    file_count += 1
    if file_count % 1000 == 0:
        print('Read file:' + str(file_count))

# 删除未使用的word
useful_vocab = [word for word in vocab if counter[word] > 0]
with open('data/vocab_useful.txt', 'w', encoding='utf-8') as f2:
    for word in useful_vocab:
        f2.write(word + '\n')

# # 统计各个word的使用频率
# frequency = {word: value / count for word, value in counter.items()}
#
# unuseful_vocab_counter = 0
# for _, value in frequency.items():
#     if value == 0:
#         unuseful_vocab_counter += 1
# print('unuseful_vocab_counter:' + str(unuseful_vocab_counter))
#
# with open('dataset/frequency.txt', 'w', encoding='utf-8') as f3:
#     for word, fre in frequency.items():
#         line = word + ':' + str(fre) + '\n'
#         f3.write(line)
