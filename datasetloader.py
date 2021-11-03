'''
save amazon dataset to .npy files
'''

import  os
print(os.getcwd())

import gzip
from nltk.tokenize import wordpunct_tokenize
import numpy as np

# 10万的数据量
# construct sub_dataset 50k: 40k train; 5k val; 5k test
# 类别不均衡，换一个产生数据集的逻辑
# import random
# seed = 42
# random.seed(seed)
# totoal_index = range(1689188)
# subset_index = random.sample(totoal_index, 50000)

# 数据集
data_path = "/home/aistudio/data/data114776/reviews_Electronics_5.json.gz"
review_text = []
review_rating = []
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

# count for 5 classes, 8k for each; train data
item_num = 8e3
val_start = 1
cnt_0,cnt_1,cnt_2,cnt_3,cnt_4 = 0,0,0,0,0
cnt = 0
for item in parse(data_path):
    if cnt < 4e4: # train set
        label = item["overall"]-1
        if item["overall"]-1 == 0:
            if cnt_0 == item_num:
                continue
            cnt_0 = cnt_0 + 1
        elif item["overall"]-1 == 1:
            if cnt_1 == item_num:
                continue
            cnt_1 = cnt_1 + 1
        elif item["overall"]-1 == 2:
            if cnt_2 == item_num:
                continue
            cnt_2 = cnt_2 + 1
        elif item["overall"]-1 == 3:
            if cnt_3 == item_num:
                continue
            cnt_3 = cnt_3 + 1
        elif item["overall"]-1 == 4:
            if cnt_4 == item_num:
                continue
            cnt_4 = cnt_4 + 1
    else:
        if val_start:
            cnt_0, cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0, 0
            item_num = 1e3
            val_start = 0
        else:
            pass
        if cnt_0==item_num and cnt_1==item_num and cnt_2==item_num and cnt_3==item_num and cnt_4==item_num:
            break
        label = item["overall"]-1
        if item["overall"]-1 == 0:
            if cnt_0 == item_num:
                continue
            cnt_0 = cnt_0 + 1
        if item["overall"]-1 == 1:
            if cnt_1 == item_num:
                continue
            cnt_1 = cnt_1 + 1
        if item["overall"]-1 == 2:
            if cnt_2 == item_num:
                continue
            cnt_2 = cnt_2 + 1
        if item["overall"]-1 == 3:
            if cnt_3 == item_num:
                continue
            cnt_3 = cnt_3 + 1
        if item["overall"]-1 == 4:
            if cnt_4 == item_num:
                continue
            cnt_4 = cnt_4 + 1
    cnt = cnt + 1
    review_text.append(wordpunct_tokenize(item["reviewText"]))
    review_rating.append(label)

# construct word_dict
word_dict = {'PADDING': 0}
for sent in review_text:
    for token in sent:
        if token not in word_dict:
            word_dict[token] = len(word_dict)

#  sentence 2 word idxes
news_words = []
for sent in review_text:
    sample = []
    for token in sent: # 把一个句子转换为对应 word 的 idx 序列，句子最多 512 个词
        sample.append(word_dict[token])
    sample = sample[:512] # 超出的部分截取掉
    news_words.append(sample + [0] * (512 - len(sample))) # 不足的部分补零

data = np.array(news_words, dtype='int32')
label = np.array(review_rating, dtype='int32')

# save it to .npy files
np.save("./dataset/data.npy",data)
np.save("./dataset/label.npy",label)
np.save("./dataset/num_tokens",len(word_dict))