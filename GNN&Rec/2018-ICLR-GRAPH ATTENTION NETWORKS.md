# 图注意力网络Graph Attention Networks

> https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1710.10903

## Abstract

本文提出了一种新的应用于图数据上的网络结果**Graph Attention Networks(GATs)**，不同与先前一些基于谱域的图神经网络，通过使用masked的self attention，从而克服了先前图卷积网络方面的短板。**GATs**既能够解决**inductive problems**，也能解决**transductive problems**。最终的实验结果也表明**GATs**在实际应用中有很好的效果。
这里插播一下**inductive problems**和**transductive problems**：
主要区别是在，训练阶段是否利用了Test data。Inductive learning对未来的Testing data并不知，Transductive learning则是已知Testing data的，当然Testing data是unlabeled。

如何理解 inductive learning 与 transductive learning?

<https://www.zhihu.com/question/68275921>

个人理解是，对于GCNs这样的方法，在训练时会用到整个graph的信息，相当于是Transductive learning。

## Introduction

这部分分析了从CNN到GNN的发展历程，然后分别剖析了现有基于谱域的GNN方法和非谱域GNN方法的不足之处。
首先，基于谱域的GNN不断进化，通过在傅利叶定义域内对图上拉普拉斯算子进行特征分解来实现卷积操作，再到引入平滑参数的filter，再到通过图的切比雪夫展开的filter，最后是限制只对一阶邻居进行操作的方法。所有这些方法都要依靠拉普拉斯矩阵的特征值，这就必须对整个图结构进行操作，这就使得一个训练好的模型很难适用于其他问题。
对于非谱域的方法，即空域上的GNN方法，通过对空间中邻居节点进行卷积操作是一个方法，但是这需要解决如何处理不同节点不一样的邻居数目。一些方法是使用图的度矩阵作为输入，或者固定邻居节点的数量。
接着，论文介绍了一下Attention mechanism。
基于以上的工作，论文提出了**GATs**，该模型具有以下优点：

- 并行处理**节点-邻居节点**对；
- 可以处理有不同度节点；
- 该模型是inductive的，即不需要知道整个图的全部信息；

## GAT Architecture

### Graph Attention Layer

模型**GAT**通过堆叠graph attention layer实现，这里说明graph attention layer。该层的输入是节点特征集合 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D%3D%5Cleft%5C%7B%5Cvec%7Bh%7D_%7B1%7D%2C+%5Cvec%7Bh%7D_%7B2%7D%2C+%5Cldots%2C+%5Cvec%7Bh%7D_%7BN%7D%5Cright%5C%7D%2C+%5Cvec%7Bh%7D_%7Bi%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7BF%7D) ，输出是一个新的节点特征集合 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D%5E%7B%5Cprime%7D%3D%5Cleft%5C%7B%5Cvec%7Bh%7D_%7B1%7D%5E%7B%5Cprime%7D%2C+%5Cvec%7Bh%7D_%7B2%7D%5E%7B%5Cprime%7D%2C+%5Cldots%2C+%5Cvec%7Bh%7D_%7BN%7D%5E%7B%5Cprime%7D%5Cright%5C%7D%2C+%5Cvec%7Bh%7D_%7Bi%7D%5E%7B%5Cprime%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7BF%5E%7B%5Cprime%7D%7D) 。为了计算每个邻居节点的权重，通过一个**F×F'**的共享权重矩阵**W**应用于每个节点，然后即可计算出attention系数：
![[公式]](https://www.zhihu.com/equation?tex=e_%7Bi+j%7D%3Da%5Cleft%28%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bi%7D%2C+%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bj%7D%5Cright%29)
这个系数可以表示节点 **j** 相对于节点 **i** 的重要性。论文只计算节点 **i** 的邻居节点，这个称作masked attention。接着就是归一化权重系数：
![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bi+j%7D%3D%5Coperatorname%7Bsoftmax%7D_%7Bj%7D%5Cleft%28e_%7Bi+j%7D%5Cright%29%3D%5Cfrac%7B%5Cexp+%5Cleft%28e_%7Bi+j%7D%5Cright%29%7D%7B%5Csum_%7Bk+%5Cin+%5Cmathcal%7BN%7D_%7Bi%7D%7D+%5Cexp+%5Cleft%28e_%7Bi+k%7D%5Cright%29%7D)
进一步细化公式如下， **a** 向量为F'×F'，使用**LeakyReLU**(α = 0.2)：
![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bi+j%7D%3D%5Cfrac%7B%5Cexp+%5Cleft%28%5Ctext+%7B+LeakyReLU+%7D%5Cleft%28%5Coverrightarrow%7B%5Cmathbf%7Ba%7D%7D%5E%7BT%7D%5Cleft%5B%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bi%7D+%5C%7C+%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bj%7D%5Cright%5D%5Cright%29%5Cright%29%7D%7B%5Csum_%7Bk+%5Cin+%5Cmathcal%7BN%7D_%7Bi%7D%7D+%5Cexp+%5Cleft%28%5Ctext+%7B+LeakyReLU+%7D%5Cleft%28%5Coverrightarrow%7B%5Cmathbf%7Ba%7D%7D%5E%7BT%7D%5Cleft%5B%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bi%7D+%5C%7C+%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bk%7D%5Cright%5D%5Cright%29%5Cright%29%7D)
这样即可得到节点 **i** 的表示：
![[公式]](https://www.zhihu.com/equation?tex=%5Cvec%7Bh%7D_%7Bi%7D%5E%7B%5Cprime%7D%3D%5Csigma%5Cleft%28%5Csum_%7Bj+%5Cin+%5Cmathcal%7BN%7D_%7Bi%7D%7D+%5Calpha_%7Bi+j%7D+%5Cmathbf%7BW%7D+%5Cvec%7Bh%7D_%7Bj%7D%5Cright%29)
为了是self-attention能稳定地表示节点 **i** ，这里还使用了multi-head attention机制，进一步调整公式为：
![[公式]](https://www.zhihu.com/equation?tex=%5Cvec%7Bh%7D_%7Bi%7D%5E%7B%5Cprime%7D%3D%7B%7C%7C%7D_%7Bk%3D1%7D%5E%7BK%7D+%5Csigma%5Cleft%28%5Csum_%7Bj+%5Cin+%5Cmathcal%7BN%7D_%7Bi%7D%7D+%5Calpha_%7Bi+j%7D%5E%7Bk%7D+%5Cmathbf%7BW%7D%5E%7Bk%7D+%5Cvec%7Bh%7D_%7Bj%7D%5Cright%29)
当然，如果使用上式multi-head attention层直接作为输出层明显不合适，所以也可以使用**K**个head的均值，如下式：
![[公式]](https://www.zhihu.com/equation?tex=%5Cvec%7Bh%7D_%7Bi%7D%5E%7B%5Cprime%7D%3D%5Csigma%5Cleft%28%5Cfrac%7B1%7D%7BK%7D+%5Csum_%7Bk%3D1%7D%5E%7BK%7D+%5Csum_%7Bj+%5Cin+%5Cmathcal%7BN%7D_%7Bi%7D%7D+%5Calpha_%7Bi+j%7D%5E%7Bk%7D+%5Cmathbf%7BW%7D%5E%7Bk%7D+%5Cvec%7Bh%7D_%7Bj%7D%5Cright%29)
模型的直观表示见下图。

![img](https://pic3.zhimg.com/80/v2-f8593bcf924814b7b91469c05749435a_hd.jpg)

## Evaluation

实验部分使用了Cora，Citeseer，Pubmed和PPI数据集做了实验。数据集的统计描述情况如下表：

![img](https://pic4.zhimg.com/80/v2-6f9d48cc5e0449d27eb9e43dd6ab0747_hd.jpg)

对比分别从Transductive learning和Inductive learning角度对比，针对Transductive learning对比了MLP、DeepWalk、GCN、MoNet等方法，针对Inductive learning则对比了GraphSAGE及其变种方法。

### Transductive Learning 

![img](https://pic4.zhimg.com/80/v2-5e27003d2406adda1ae563e015eb3ec3_hd.jpg)

### Inductive Learning 

![img](https://pic3.zhimg.com/80/v2-ddfdd7178a1c47763a7982b4aec8945e_hd.jpg)

从对比实验可以看出，**GATs**在Transductive learning、Inductive learning中都有最好的结果。这也说明了模型的有效性。同时，论文还使用**GATs**生成的节点表示进行了t-SNE聚类来直观的展示模型的效果。

![img](https://pic3.zhimg.com/80/v2-3b0ca1e0f801d3b1564390c20f5f0032_hd.jpg)

### Conclusion

最后，对模型做了总结。

Github上也有找到GATs的实现：

<https://link.zhihu.com/?target=https%3A//github.com/PetarV-/GAT>

<https://github.com/Diego999/pyGAT>

**往期文章：**

- Gotcha - Sly Malware 基于metagraph2vec的恶意软件检测系统 <https://zhuanlan.zhihu.com/p/73011375>
- DeepWalk Online Learning of Social Representations <https://zhuanlan.zhihu.com/p/68328470>
- 基于注意力的GNN的半监督学习AGNN <https://zhuanlan.zhihu.com/p/65732106>
- 用于推荐的异构信息网络嵌入方法 HIN Embedding for Recommendation　<https://zhuanlan.zhihu.com/p/63329653>
- 异构图注意力网络 Heterogeneous Graph Attention Network 　[https://zhuanlan.zhihu.com/p/62](https://zhuanlan.zhihu.com/p/62884521)