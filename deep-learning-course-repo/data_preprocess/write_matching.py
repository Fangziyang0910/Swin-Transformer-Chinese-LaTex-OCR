import os

# 根据数据集路径改data_path, 其他参数不需要改
# data_path = '210705'

sub_name = ['train', 'val', 'test']
# formulas = '../../images/' + data_path + '/formulas/'
formulas = '../../images/formulas/'
if not os.path.isdir('../../images' + '/matching/'):
    os.mkdir('../../images' + '/matching/')

for sn in sub_name:
    match = '../../images/' + '/matching/' + sn +'.matching.txt'
    with open(formulas + sn + '.formulas.norm.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    size = len(lines)
    for i in range(size):
        with open(match, 'a', encoding='utf-8') as f:
            fileName = str(i) + '.png'
            f.write(fileName + ' ' + str(i) + '\n')