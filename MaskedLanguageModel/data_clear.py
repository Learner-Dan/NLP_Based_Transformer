#
# data_clear.py
# @author DanHe
# @description 
# @created 2021-05-26T16:31:53.755Z+08:00
# @last-modified 2021-05-29T10:54:18.574Z+08:00
#

import matplotlib.pyplot as plt
import numpy as np 
import unicodedata

data_path = "./data/wikitext-2/train.txt"

tp = open(data_path)
lines = tp.readlines()
tp.close()

# 统计词汇信息和每行句子的长度
len_statistic = []
sentense_len = []
word_statistic = {}
for line in lines:
    line = " ".join(line.strip().split())
    line_split = line.split()
    for l in line_split:
        if l not in word_statistic.keys():
            word_statistic[l] = 0
        word_statistic[l] += 1
    num_str = len(line)
    sentense_len.append(len(line_split))
    len_statistic.append(len(line))
print("Total words is %d"%len(word_statistic))
print("max row len is : %d, min row len is %d"%(np.max(len_statistic),np.min(len_statistic)))
print("max sentense len is: %d,min sentense len is: %d"%(np.max(sentense_len),np.min(sentense_len)))

# 绘制长度直方图
hist = plt.hist(len_statistic,bins=40)
plt.show()
print(hist)

# 对词汇按照出现频率排序
word_num = []
word_info = []
for key,value in word_statistic.items():
    word_num.append(value)
    word_info.append(key)
word_num = np.array(word_num)
word_sort_index = np.argsort(-word_num)
word_num = word_num[word_sort_index]
word_info = [word_info[i] for i in word_sort_index]


# 过滤过短的句子及特殊符号
save_lines = []
for line in lines:
    line = line.strip()
    line_split = line.split()
    new_line = ""
    word_num = 0
    for l in line_split:
        if l == "":
            continue
        elif l== "=":
            continue
        elif (l == ".") | (l == ";"):
            new_line.strip()
            if(word_num >= 10 | (len(new_line)>10)):
                save_lines.append(new_line)
                new_line = ""
                word_num = 0
                continue
        new_line = new_line + " " + l
        word_num += 1
        if word_num>=300:
            new_line = new_line.strip()
            # new_line = unicodedata.normalize('NFD', new_line)
            # ''.join(c for c in new_line if not unicodedata.combining(c))
            new_line = new_line.lower()
            save_lines.append(new_line)
            new_line = ""
            word_num = 0
            continue
    new_line = new_line.strip()
    new_line = new_line.lower()
    # new_line = unicodedata.normalize('NFD', new_line)
    # ''.join(c for c in new_line if not unicodedata.combining(c))
    if(word_num >= 10 | (len(new_line)>10)):
        save_lines.append(new_line)

save_path = "./data/wikitext-2/train.txt"
tp = open(save_path,"w")
for line in save_lines:
    tp.write("%s\n"%line)
tp.close()