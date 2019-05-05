# ATTENTION-BASED GRAPH NEURAL NETWORK FOR SEMI-SUPERVISED LEARNING

[TOC]

> 这是一篇发表在ICLR2018的文章。

## ABSTRACT

近段时间，graph neural networks（GNN）在一系列公开的基于图的半监督任务中表现出了不俗的性能。那些架构都是交替使用**融合邻居节点隐藏状态的传播层**和**全连接层**。本文提出的一种去除所有中间的全连接层的线性模型的表现依旧能与最优的模型媲美。这个模型能能减少参数量，这对样本较少的半监督学习任务来说具有重要意义。这种新的GNN同时也结合了注意力机制。

## INTRODUCTION



