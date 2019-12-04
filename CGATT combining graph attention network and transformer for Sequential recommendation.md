# CGATT: Combining Graph Attention Network and Transformer for Sequential Recommendation

## Abstract



## Introduction



## Related work

Kang et al. proposed self-attention based sequential model SASRec that seeks to balance the ablities of  Recurrent Neural Networks and Markov Chains.

Ying et al. proposed a novel two-layer hierarchical attention network model SHAN to capture the dynamic  property of users.

HNVM 



## Preliminary



## The recommendation model

In this part, we 

### Problem statement



###  Making recommendation

After  obtained  the  unified  representation  of  user $u$,  we compute  the  score  $\hat{z_i}$ for  each  candidate  item $v_i \in V$ by multiplying its embedding  $v_{i}$  by the user representation $S$, which can be defined as:

$$\hat{z}_{i}=\mathrm{z}_{u}^{\top} e_{v_{i}}.$$

 Then we apply a softmax function to get the output vectorof the model:

$$\mathrm{\hat{y}} = softmax{(\hat{z})},$$

where $\hat{z} \in R^{|V|}$ denotes the recommendation scores over allcandidate  items  and  $\hat{y}$ denotes  the  probabilities  that  items will be interacted by user $u$ next time in the current session $S_c^u$. 

For  any  given  user  behavior  graph,  the  loss  functionis  defined  as  the  cross-entropy  of  the  prediction  and  theground truth. It can be written as follows: 

$$\mathcal{L}(\hat{\mathbf{y}})=-\sum_{i=1}^{|V|} y_{i} \log \left(\hat{y}_{i}\right)+\left(1-y_{i}\right) \log \left(1-\hat{y}_{i}\right),$$

where $y$ denotes the one-hot encoding vector of the groundtruth  item.  Finally,  we  use  the  back-propagation  through time   (BPTT)   algorithm   to   train   the   proposed   A-PGNN method. 

## Experimental results and analysis

In this section we describe the experimental setup and take a discussion of the result obtained from the model **CGATT**.

### Datesets

We evaluate the proposed model on four real-world representative datasets which vary significantly in domains and sparsity.

* **MovieLens**: This is a popular benchmark dataset for evaluating recommendation algorithms. In this work, we adopt two well-established versions, **MovieLens 100k (ml-100k)** and **MovieLens 1m** **(ml-1m)**.
* **Amazon Office**: This is a series of product review datasets crawled from Amazon.com by McAuley et al..

###  Evaluation metrics 



###   Baselines

To validate the effecitveness of our model **CGATT**, we compare our model with the following sequential recommendation methods:

* **POP**: It is the simplest baseline that ranks items according to their popularity judged by the number of interactions.
* **GRU4Rec**: It uses GRU with ranking based loss to model user sequences for session based recommendation.
* **GRU4Rec+**: It is an improved version of GRU4Rec with a new class of loss functions and sampling strategy.
* **Caser**: It employs CNN in both horizontal and vertical way to model high-order MCs for sequential recommendation.
* **SASRec**: It uses a left-to-right Transformer language model to capture users’ sequential behaviors, and achieves state-of-the-art performance on sequential recommendation.



### Detail analysis

#### Sequential length

In order to understand the scalability of sequential recommendationmodels when applied with sessions of different lengths,  we dividethe sessions in the datasets into *short* ( no more than 5 queries ),  *medium* ( 6 to 15 queries ) and *long* ( more than 15 queries ) on the test set and report separate results in Fig. 4. We do not Item-pop and FPMC in the comparison, as their performance is worsethan that of theRNN-based models, especially with short sessions. 

####  Influence of  laysers



### Case study



## Conclusions

In this paper, we proposed a novel graph attention network and transformer based model **CGATT** for recommending next item. Not only utilize we transformer encoder to capture the sequential information of user behavior, but also consider the complex realtion information of consumed items. From the extensive experiments, our model outperforms state-of-the-art methods for sequential recommendation on vatiety real-world datasets. 



## Reference

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
# 评估方法亦可参照上文。
```

```
@inproceedings{ying2018sequential,
  title={Sequential recommender system based on hierarchical attention networks},
  author={Ying, Haochao and Zhuang, Fuzhen and Zhang, Fuzheng and Liu, Yanchi and Xu, Guandong and Xie, Xing and Xiong, Hui and Wu, Jian},
  booktitle={the 27th International Joint Conference on Artificial Intelligence},
  year={2018}
}
```

```
@inproceedings{chen2019dynamic,
  title={A Dynamic Co-attention Network for Session-based Recommendation},
  author={Chen, Wanyu and Cai, Fei and Chen, Honghui and de Rijke, Maarten},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={1461--1470},
  year={2019},
  organization={ACM}
}
```

```
@inproceedings{zhang2018deep,
  title={A deep joint network for session-based news recommendations with contextual augmentation},
  author={Zhang, Lemei and Liu, Peng and Gulla, Jon Atle},
  booktitle={Proceedings of the 29th on Hypertext and Social Media},
  pages={201--209},
  year={2018},
  organization={ACM}
}
```