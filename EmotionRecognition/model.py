from transformers import BertModel
from torch import nn

class BERTClassifier(nn.Module):

    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self, output_dim, pretrained_name='bert-base-chinese'):
        super(BERTClassifier, self).__init__()

        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained(pretrained_name)

        # 外接全连接层
        self.mlp = nn.Linear(768, output_dim)

    def forward(self, tokens_X):
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        res = self.bert(**tokens_X)

        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析
        return self.mlp(res[1])
