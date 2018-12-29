---
layout: post
title:  "如何清洗人工标注的错误数据"
date:   2018-10-06 15:40:56
categories: machine learning
---

## 背景

做机器学习的小伙伴应该都有感触，训练集是否足够干净，覆盖性是否足够广，是某个模型算法效果好不好的重要因素。然而我们工程生产中的训练集大多来自爬虫，众包，人工标注，其中不免有很多脏数据。实际经验中，有的训练集的脏数据占比甚至超过了10%，这对我们选择合适的线上模型带来了很大噪声，让每个工程师头疼不已。

笔者提出了一些自动化和半自动化清洗训练集的思路和工具，希望帮助广大机器学习从业者对自己手中的训练集尝试做一些修改和提升。

## 实现原理

对于普适化的机器学习任务，输入是模型架构和训练集，输出是预测值。训练集本身会含有一些的错误标注case，我们的目的就是找出这些标注有错的case并尝试替换成正确的。

一种比较通用的思想就是利用多种模型的 ensemble 投票结果，以 k-folds的方式对每块训练集做预测，即模型通过剩下的 80% 训练集训练的结果来预测剩下的20%的case，如果多个模型的预测结果都相同而且跟训练集的标注结果不符，那么我们有理由认为这条case跟大众标注结果不符，可能需要用预测结果替换掉。

基本逻辑如下：

```
1. 首先针对当前任务 T，选择可以完成 T 的 M 个模型备用
2. 将训练集拆成 k part
3. foreach part
	以当前part之外的数据作为训练集，分别训练上述 M 个模型，对当前 part 预测结果。
4. k折训练之后，训练集中的每一条case都拥有了 标注值 和 预测值List
5. 设计 Reward 表示根据预测值List 对标注值替换是否有增益。
6. 给定阈值，对Reward 大于阈值的case 完成最优结果替换，生成新的训练集。
```

我们以NLP中的命名实体识别（NER）任务为例说明整个流程的详细过程，过程中用python作为伪代码主要讲解思路。

NER 目前效果比较不错的模型有 CRF，Bilstm-CRF等，这里仅提供demo思路，因此只使用了CRF和Bilstm-CRF两个模型，通过设置不同的参数初始化出不同的模型，作为评价的模型池。

```python
cleaner = Cleaner()
# add model
cleaner.add_model(CRFModel())
cleaner.add_model(BiLstmCRFModel(embed_dim=100, rnn_unit=180))
cleaner.add_model(BiLstmCRFModel(embed_dim=200, rnn_unit=300))
```

然后我们加载数据集，并完成 k-folds 拆分：

```python
kf = KFold(n_splits=args.k, shuffle=True, random_state=119)
for train_index, test_index in kf.split(dataset):
	# Do training models, 在 do_model_train 中，我们可以完成对 model 的训练和预测，构建 预测值 List。
	preds = []
	for m in clean.models:
		m.fit(dataset.get_index_of(train_index))
		pred = m.predict(dataset.get_index_of(test_index))
		preds.append(pred)
```

完成预测之后，我们需要对 preds 和 dataset.label 计算替换增益 Reward。

```python
scores = cal_reward(preds, dataset.label)
```

Scores 在 NER 中的计算规则如下：

```
预测值与标注值不同 score += 1，并计入 error_list
模型预测值与标注不同的 list 中，error_list中模型预测值如果有相同的，score += 相同个数 * lambda
如果模型预测值中的实体个数有所减少，则对score -= 100
```

然后根据阈值过滤出需要更新的case，完成对训练集的替换即可。

## 效果展示

我们借助在某实际项目中遇到的 NER 训练集，完成本文思路在实际项目的落地评测。此NER Task针对于 Sports 领域做命名实体识别，识别目标包括 球队，球员，时间，场次等等。

在规模 30000 条左右的训练集中，运用本文方法做了自动替换，替换条数 1463 条，并且用实际线上采集标注的 3945 条数据做为测试集，测试集上的表现如下：

| | 原始训练集  | 替换后的训练集  |
| ------------ | ------------ | ------------ |
| 全优准确率  | 2550/3945  | 2609/3945  |

## 优化方向

#### 加入人工判断

- 在完成替换之前，导出即将替换的case，找标注同学过滤一遍，可以对更新的操作有进一步的提升。

#### 多次执行

- 可以依据第一个版本的训练集再次执行，可以考虑一定程度放宽阈值。

#### 针对 NER 任务的优化思路

- 借助 NER 训练集可以统计出来 某个词汇标注为某个实体标签 的先验概率，在计算 替换 Reward 的时候，将标签替换概率转移得分也添加到 score 计算过程中。
- 人工 check 预测 list 和标注值不符，但是无法判断是否正确的时候，可以将训练集包含相同实体的case找出来，大概率是实体标注标准不一致导致的，可以手动改正，样本量不足的还可以新增训练样本。

## 附录

本文相关源码：[待开源]。


