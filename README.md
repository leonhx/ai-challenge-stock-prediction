# 虚拟股票趋势预测

比赛网站：[https://challenger.ai/competition/trendsense](https://challenger.ai/competition/trendsense)

## TODO

1. 实现交叉验证
1. `group`字段做embedding
1. 加上tensorboard summary打点

## 思路

1. 聚类
1. `group`和`era`字段应该都比较重要，尤其是`group`字段（测试集里也有此字段）
1. 特征可能是时序的
1. pseudo labeling
