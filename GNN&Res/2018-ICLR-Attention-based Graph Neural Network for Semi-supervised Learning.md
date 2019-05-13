# ATTENTION-BASED GRAPH NEURAL NETWORK FOR SEMI-SUPERVISED LEARNING

[TOC]

> 这是一篇发表在ICLR2018的文章。

## ABSTRACT

近段时间，graph neural networks（GNN）在一系列公开的基于图的半监督任务中表现出了不俗的性能。那些架构都是交替使用**融合邻居节点隐藏状态的传播层**和**全连接层**。本文提出的一种去除所有中间的全连接层的线性模型的表现依旧能与最优的模型媲美。这个模型能能减少参数量，这对样本较少的半监督学习任务来说具有重要意义。这种新的GNN同时也结合了注意力机制。

## INTRODUCTION

这篇论文关注点在于那些图结构中的未标记的数据。一个图包含数据点之间本质的关系，而这些关系不一定都是标记好的。论文拿一个引文网络举例，图中节点是有bag-of-words的feature vector表示，每条edge表示一个引用关系，同样也会隐含一种作者和其他引文之间的关系，这些关系不能由单独的bag of words 的feature vector推断出来。例如社交网络、购物记录、观影历史，都有这样的性质，特征向量不足以表示图结构所富含的丰富意义。基于图的半监督学习任务是使用一小部分标记节点和所有节点特征来对节点分类。GNN作为一种新的神经网络在一系列标准数据集中有重要的提升。这篇论文提出使用注意力机制来聚合邻居信息。论文模型主要有三方面优势：

* 极大减少了模型复杂度；
* 动态自适应发现节点之间关系；
* 进一步提高了在标准数据集上的准确率；

## RELATED WORK

### Graph Laplacian regularization

一般的，一个图中的临近节点更有可能有同样的label，整个图的信息可以这样表示：
$$
\mathcal{L}\left(X, Y_{L}\right)=\mathcal{L}_{\text { label }}\left(X_{L}, Y_{L}\right)+\lambda \mathcal{L}_{G}(X)
$$
其中L<sub>G</sub>是基于图的正则项，称为拉普拉斯正则化。

### Unsupervised node embedding for semi-supervised learning

目前已经有几种使用连接来做隐式欧几里德空间的node embedding。通过学习到的embedding，就可以通过标准的监督学习来训练模型。word2vec、DeepWalk、node2vec、LINE等等。这些方法的优势在于其普遍性，这也是它们针对特定任务的缺陷所在。

### Graph Neural Network (GNN)

GNN是神经网络图结构化的扩展，或者说是CNN、RNN的一种扩展。通常，信息聚合是依据一些神经网络结构迭代进行的。模型参数通过监督、半监督方式的来训练的。实际上，GNN已经得到了广泛的应用。论文接下来先会先介绍GCN来对比论文方法。

## DISSECTION OF GRAPH NEURAL NETWORK

论文提出一种新的基于GNN的模型，*Attention-based Graph Neural Network* (**AGNN**)。模型**Z**的定义如下：
$$
Z=f(X, A)
$$
其中，**X**是节点特征，**A**是graph的邻接矩阵。

通常，典型的GNN模型前向通道是传播层和单层感知器交替使用。一个传播矩阵**P**的传播层可以这样定义：
$$
\tilde{H}^{(t)}=P H^{(t)}
$$
其中，
$$
P=D^{-1} A  ，\tilde{H}_{i}^{(t)}=(1 /|N(i)|) \sum_{j \in N(i)} H_{j}^{(t)}，D=\operatorname{diag}(\mathrm{A} \mathbb{1})
$$
