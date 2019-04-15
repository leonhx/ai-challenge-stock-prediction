# 虚拟股票趋势预测

比赛网站：[https://challenger.ai/competition/trendsense](https://challenger.ai/competition/trendsense)

## TODO

1. experiment接口：https://mp.weixin.qq.com/s/qGe37OY_Iy9Qr9Rabpv7CA
1. 实现交叉验证
1. `group`字段做embedding
1. 加上tensorboard summary打点
1. 结果的稳定性（数据量小）

## 思路

1. 聚类
1. `group`和`era`字段应该都比较重要，尤其是`group`字段（测试集里也有此字段）
1. 特征可能是时序的
1. pseudo labeling
1. RNN
1. Attention
1. CNN底层特征 + RNN
1. DenseNet (Densely Connected Convolutional Networks，ResNet的求和换成concat)
1. Bottleneck (of CNN, mentioned in fastai course before,上面那篇论文里也用到了，好像是1 * 1卷积、减少维数、增加filter数量)
1. TensorBoard监控中间隐藏层的值、梯度，看是否需要加batch norm
1. 最后不要dense layer，直接global average pooling试试看

## 小数据集上结果的稳定性

1. Dropout
1. Bagging
1. Contractive AutoEncoder预训练 + 微调
1. VAE预训练 + 微调
1. Jacobian矩阵F范数作为正则项
1. CAE + VAE
1. 原始数据加入噪声（目的：Data Augmentation或者提高模型鲁棒性）
1. 权重加噪声
1. 加入`group`字段时，最好在没有加这个字段的基础上微调
1. 贝叶斯模型在小数据集上表现比较好
1. 置信区间比点估计好
1. 论文：Improving neural networks by preventing co-adaptation of feature detectors
1. 论文：Training Neural Networks with Very Little Data -- A Draft
