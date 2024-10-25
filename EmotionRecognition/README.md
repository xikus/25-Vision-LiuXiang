![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

### 语句情感识别 

- **模型架构**
    >本模型基于 *bert* 模型修改而来，主要是在 *bert* 模型的基础修改了 *output* ，用于分类任务。

- **代码思路**
    >1. 每次训练时，将数据集中的数据进行 *tokenize* 处理，然后将处理后的数据输入到 *bert* 模型中，得到 *output*。
    >2. 批量训练，每次训练8条样本数据(batchSize = 8)
    >3. 为避免训练时间过长, 本模型只采用`简单语句.csv`作为数据集

- **函数**
  ```python
  read_file() # 读取训练集中的评论数据和对应的标签信息
  evaluate() # 评估模型的准确率
  train_bert_classifie() # 训练模型
  ```

- **训练效果**
```Bash
--train_acc: tensor(0.9692, device='cuda:0') 	--test_acc tensor(0.9894, device='cuda:0')
```

- **文件结构**
```Bash
├── bert-base-chinese
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── bert.parameters
├── model.py
├── __pycache__
│   └── model.cpython-39.pyc
├── README.md
├── recognize.py
└── SentenceAnalysis
    ├── 简单语句.csv
    ├── 豆瓣影评.csv
    └── 豆瓣影评(部分).csv

# bert-base-chinese: 预训练的bert模型和tokenizer
# SentenceAnalysis: 训练集
# recognize.py: 训练程序
# model.py: 模型代码
```
- **使用方法**
```Bash
python3 recognize.py
```

