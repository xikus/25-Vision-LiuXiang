import csv
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm

"""
读取评论文件的评论信息
"""


def read_file(file_name):
    comments_data = []

    # 读取评论信息
    with open(file_name, 'r', encoding='GB18030') as f:
        reader = csv.reader(f)

        #跳过第一行
        next(reader)

        # 读取评论数据和对应的标签信息
        for line in reader:
            if len(line[0]) > 0:
                if line[1] == 'positive':
                    comments_data.append([line[0], 0])
                if line[1] == 'negative':
                    comments_data.append([line[0], 1])
                if line[1] == 'neutral':
                    comments_data.append([line[0], 2])

        # 打乱数据集
    random.shuffle(comments_data)

    data = pd.DataFrame(comments_data)

    same_sentence_num = data.duplicated().sum()  # 统计重复的评论内容个数

    if same_sentence_num > 0:
        data = data.drop_duplicates()  # 删除重复的样本信息

    f.close()

    return data


comments_data = read_file('./简单语句.csv')
split = 0.6
split_line = int(len(comments_data) * split)

# 划分训练集与测试集，并将pandas数据类型转化为列表类型
train_comments, train_labels = list(comments_data[: split_line][0]), list(comments_data[: split_line][1])
test_comments, test_labels = list(comments_data[split_line:][0]), list(comments_data[split_line:][1])

