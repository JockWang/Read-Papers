# 基于图卷积网络的半监督分类Semi-Supervised Classification with Graph Convolutional Networks

> 这篇文章发表于ICLR2017，作者：Thomas N. Kipf & Max Welling。

## ABSTRACT

论文提出了一种基于图结构数据的半监督学习的方法，是直接在图上操作的卷积神经网络的变种。

## INTRODUCTION

图结构中的节点分类一般都是依靠少量的带有标签的节点的子集来完成。这类问题可以看作是图结构的半监督学习，标签信息通过一些精确到基于图的正则的形式均匀的分布在图上，比如在损失函数中使用图拉普拉斯正则化项：
$$
\mathcal{L}=\mathcal{L}_{0}+\lambda \mathcal{L}_{\mathrm{reg}}, \quad \text { with } \quad \mathcal{L}_{\mathrm{reg}}=\sum_{i, j} A_{i j}\left\|f\left(X_{i}\right)-f\left(X_{j}\right)\right\|^{2}=f(X)^{\top} \Delta f(X)
$$
这里的$\mathcal{L}_0$表示监督的损失，即图中有标签部分，