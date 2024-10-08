import csv
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
import model

path ='/home/yoda/25-Vision-LiuXiang/EmotionRecognition/bert-base-chinese'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU，有则使用GPU，否则使用CPU
#device = d2l.try_gpu()  # 尝试使用GPU

net = model.BERTClassifier(output_dim=3,pretrained_name=path)                      # BERTClassifier分类器，因为最终结果为3分类，所以输出维度为3，代表概率分布
net = net.to(device)                                    # 将模型存放到GPU中，加速计算

# 定义tokenizer对象，用于将评论语句转化为BertModel的输入信息
tokenizer = BertTokenizer.from_pretrained(path)

loss = nn.CrossEntropyLoss()                                # 损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)      # 小批量随机梯度下降算法

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

    data = pd.DataFrame(comments_data)  # 将数据转化为pandas数据类型

    same_sentence_num = data.duplicated().sum()  # 统计重复的评论内容个数

    if same_sentence_num > 0:
        data = data.drop_duplicates()  # 删除重复的样本信息

    f.close()

    return data


comments_data = read_file('SentenceAnalysis/简单语句.csv')
split = 0.6
split_line = int(len(comments_data) * split)

# 划分训练集与测试集，并将pandas数据类型转化为列表类型
train_comments, train_labels = list(comments_data[: split_line][0]), list(comments_data[: split_line][1])
test_comments, test_labels = list(comments_data[split_line:][0]), list(comments_data[split_line:][1])


"""
评估函数，用以评估数据集在神经网络下的精确度
"""


def evaluate(net, comments_data_, labels_data):
    sum_correct, i = 0, 0

    while i <= len(comments_data_):
        comments = comments_data_[i: min(i + 8, len(comments_data_))]

        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)

        res = net(tokens_X)  # 获得到预测结果

        y = torch.tensor(labels_data[i: min(i + 8, len(comments_data_))]).reshape(-1).to(device=device)

        sum_correct += (res.argmax(axis=1) == y).sum()  # 累加预测正确的结果
        i += 8

    return sum_correct / len(comments_data_)  # 返回(总正确结果/所有样本)，精确率


"""
训练bert_classifier分类器
"""


def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels,
                          device, epochs):
    max_acc = 0.5  # 初始化模型最大精度为0.5

    # 先测试未训练前的模型精确度
    train_acc = evaluate(net, train_comments, train_labels)
    test_acc = evaluate(net, test_comments, test_labels)

    # 输出精度
    print('--epoch', 0, '\t--train_acc:', train_acc, '\t--test_acc', test_acc)

    # 累计训练18万条数据 epochs 次，优化模型
    for epoch in tqdm(range(epochs)):

        i, sum_loss = 0, 0  # 每次开始训练时， i 为 0 表示从第一条数据开始训练

        # 开始训练模型
        while i < len(train_comments):
            comments = train_comments[i: min(i + 8, len(train_comments))]  # 批量训练，每次训练8条样本数据

            # 通过 tokenizer 数据化输入的评论语句信息，准备输入bert分类器
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)

            # 将数据输入到bert分类器模型中，获得结果
            res = net(tokens_X)

            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i + 8, len(train_comments))]).reshape(-1).to(device=device)

            optimizer.zero_grad()  # 清空梯度
            l = loss(res, y)  # 计算损失
            l.backward()  # 后向传播
            optimizer.step()  # 更新梯度

            sum_loss += l.detach()  # 累加损失
            i += 8  # 样本下标累加

        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels)
        test_acc = evaluate(net, test_comments, test_labels)

        # 输出精度
        print('\n--epoch', epoch + 1, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc,
              '\t--test_acc', test_acc)

        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            # 更新历史最大精确度
            max_acc = test_acc

            # 保存模型
            torch.save(net.state_dict(), 'bert.parameters')


train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels, device, 20)
