<p align="center">
 <img src="assets/deepsklearn-banner.png">
</p>
<h1 align="center">DeepSklearn</h1>
<h3 align="center">All You Need Is DeepSklearn.</h3>
<p align="center">An industrial-grade PyTorch platform for modern deep learning algorithms</p>


<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch">
  </a>
  <a href="https://github.com/guopanjin/deepsklearn/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/guopanjin/deepsklearn" alt="License">
  </a>
  <a href="https://github.com/guopanjin/deepsklearn">
    <img src="https://img.shields.io/badge/Version-1.0.0-orange" alt="Version">
  </a>
</p>

## Table of Contents
- [What is DeepSklearn?](#what-is-deepsklearn)
- [Training Dashboard](#training-dashboard)
- [Models](#models)
  - [1. Non-Sequence Models](#1-non-sequence-models)
  - [2. Sequence — User Behavior Sequence Models](#2-sequence--user-behavior-sequence-models)
  - [3. Multi-Task Learning](#3-multi-task-learning)
  - [4. Generative Models](#4-generative-models)
  - [5. Recall](#5-recall)
- [Roadmap](#roadmap)
- [Contact](#contact)
- [Citation](#citation)

## What is DeepSklearn?
DeepSklearn is an industrial-grade PyTorch platform that makes modern deep learning algorithms easy to build, train, evaluate, and deploy in a clean, sklearn-style workflow.

It combines a modular deep learning framework with an interactive Streamlit training dashboard, allowing users to configure datasets, select models, tune hyperparameters, start training, and monitor logs directly from a UI. With DeepSklearn, even beginners can launch deep learning training jobs without writing complex boilerplate training code.

The project is designed to support a growing algorithm zoo, including recommendation models, recall models, multi-task learning models, sequence models, and general neural network architectures. DeepSklearn emphasizes practical engineering, clean abstractions, reusable components, and production-inspired training pipelines.
```python
from deepsklearn.features import FeaturePipeline
from deepsklearn.datasets import TorchStreamingDataset
from deepsklearn.models import DeepFM
from deepsklearn.trainer import DiscriminativeTrainer

feature_columns = FeaturePipeline(feature_config).get_feature_columns()
model = DeepFM(feature_columns=feature_columns)

trainer = DiscriminativeTrainer(
    model_name="deepfm", model=model,
    train_dataloader=train_loader,
    validation_dataloader=val_loader)
trainer.train()
```
## training dashboard
![deepsklearn demo](assets/deepsklearn.gif)

## Models

DeepSklearn ships a growing algorithm zoo across five core domains: non-sequence ranking, user behavior sequence, multi-task learning, generative retrieval, and recall.

Every result below was produced with the **same data pipeline, feature configuration, and trainer** — differences reflect model architecture only. All runs are reproducible via the corresponding script in [`examples/`](examples/).

---

### 1. Non-Sequence Models

Benchmarked on **Criteo** — 36.6M train / 4.58M validation samples, single-machine CPU, batch size 20000, 1 epoch. Sorted by validation AUC.

| Model           | Full Name                                                                 | Paper                                                                                                                                                                    | Val AUC ↑  | Val LogLoss ↓ |
| :-------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------: | :-----------: |
| **PNN**         | Product-based Neural Network                                              | [Qu et al., 2016] [*Product-based Neural Networks for User Response Prediction*](https://arxiv.org/abs/1611.00144)                                                        | **0.8103** |  **0.4399**   |
| **DCN-V2**      | Deep & Cross Network V2                                                   | [Wang et al., 2021] [*DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*](https://arxiv.org/abs/2008.13535)             |   0.8095   |    0.4406     |
| **DLRM**        | Deep Learning Recommendation Model                                        | [Naumov et al., 2019] [*Deep Learning Recommendation Model for Personalization and Recommendation Systems*](https://arxiv.org/abs/1906.00091)                            |   0.8077   |    0.4424     |
| **DeepFM**      | Deep Factorization Machine                                                | [Guo et al., 2017] [*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*](https://arxiv.org/abs/1703.04247)                                         |   0.8072   |    0.4430     |
| **DNN**         | Deep Neural Network                                                       | —                                                                                                                                                                        |   0.8059   |    0.4439     |
| **Wide & Deep** | Wide & Deep Learning                                                      | [Cheng et al., 2016] [*Wide & Deep Learning for Recommender Systems*](https://arxiv.org/abs/1606.07792)                                                                  |   0.8055   |    0.4445     |
| **DCN**         | Deep & Cross Network                                                      | [Wang et al., 2017] [*Deep & Cross Network for Ad Click Predictions*](https://arxiv.org/abs/1708.05123)                                                                  |   0.8047   |    0.4452     |
| **NFM**         | Neural Factorization Machine                                              | [He & Chua, 2017] [*Neural Factorization Machines for Sparse Predictive Analytics*](https://arxiv.org/abs/1708.05027)                                                    |   0.8038   |    0.4458     |
| **FM**          | Factorization Machines                                                    | [Rendle, 2010] [*Factorization Machines*](https://ieeexplore.ieee.org/document/5694074)                                                                                  |   0.7997   |    0.4501     |
| **AutoInt**     | Automatic Feature Interaction Learning via Self-Attentive Neural Networks | [Song et al., 2019] [*AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*](https://arxiv.org/abs/1810.11921)                             |   0.7970   |    0.4517     |
| **AFM**         | Attentional Factorization Machine                                         | [Xiao et al., 2017] [*Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks*](https://arxiv.org/abs/1708.04617)         |   0.7948   |    0.4537     |
| **LR**          | Logistic Regression                                                       | [Cox, 1958] [*The Regression Analysis of Binary Sequences*](https://www.jstor.org/stable/2983890)                                                                        |   0.7929   |    0.4556     |

---

### 2. Sequence — User Behavior Sequence Models

Benchmarked on **Amazon Reviews — Beauty** (5-core, 2014). Sorted by validation AUC.

| Model               | Full Name                             | Paper                                                                                                                            | Val AUC ↑  | Val LogLoss ↓ |
| :------------------ | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------- | :--------: | :-----------: |
| **DIN**             | Deep Interest Network                 | [Zhou et al., 2018] [*Deep Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/1706.06978)                | **0.7275** |    0.7393     |
| **Average Pooling** | Average Pooling for Sequence Modeling | Common sequence aggregation baseline; no single canonical recommendation paper                                                   |   0.7263   |  **0.6650**   |
| **BST**             | Behavior Sequence Transformer         | [Chen et al., 2019] [*Behavior Sequence Transformer for E-commerce Recommendation in Alibaba*](https://arxiv.org/abs/1905.06874) |   0.6934   |       —       |

> On this dataset, attention-based models do not outperform simple average pooling — a known outcome on small-scale benchmarks where sequence length and data volume are insufficient to train attention effectively.

---

### 3. Multi-Task Learning

Benchmarked on **AliExpress-NL** (multi-task CTR / CVR dataset) — 36.4M train / 5.56M validation samples. Sorted by CTR AUC.

| Model             | Full Name                         | Paper                                                                                                                                                                | CTR AUC ↑  | CTCVR AUC ↑ | Val Loss ↓ |
| :---------------- | :-------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :---------: | :--------: |
| **PLE**           | Progressive Layered Extraction    | [Tang et al., 2020] [*Progressive Layered Extraction: A Novel Multi-Task Learning Model for Personalized Recommendations*](https://doi.org/10.1145/3383313.3412236) | **0.7214** | **0.8542**  | **0.1145** |
| **MMoE**          | Multi-gate Mixture-of-Experts     | [Ma et al., 2018] [*Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*](https://dl.acm.org/doi/10.1145/3219819.3220007)          |   0.7171   |   0.8527    |   0.1149   |
| **ESMM**          | Entire Space Multi-Task Model     | [Ma et al., 2018] [*Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate*](https://arxiv.org/abs/1804.07931)              |   0.7165   |   0.8525    |   0.1148   |
| **Shared Bottom** | Shared-Bottom Multi-Task Learning | [Caruana, 1997] [*Multitask Learning*](https://doi.org/10.1023/A:1007379606734)                                                                                      |   0.7152   |   0.8330    |   0.1150   |

> Per-task AUC is reported because aggregate loss masks task-level differences — all four models fall within 0.0005 on total loss, yet differ by 1.4% on CTR AUC and 2.1% on CTCVR AUC.
> The ranking follows the expected progression Shared Bottom → ESMM → MMoE → PLE, with gated expert architectures showing the clearest gains on CTCVR.

---

### 4. Generative Models

Benchmarked on **Amazon Reviews — Beauty** (5-core, 2014), identical sequence length and training configuration across all models. Sorted by validation loss.

| Model        | Full Name                                                                  | Paper                                                                                                                                                                            | Val Loss ↓ |
| :----------- | :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: |
| **HSTU**     | Hierarchical Sequential Transduction Units                                 | [Zhai et al., 2024] [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152)              | **7.1439** |
| **GRU4Rec**  | Gated Recurrent Unit for Recommendation                                    | [Hidasi et al., 2016] [*Session-based Recommendations with Recurrent Neural Networks*](https://arxiv.org/abs/1511.06939)                                                         |   7.3063   |
| **SASRec**   | Self-Attentive Sequential Recommendation                                   | [Kang & McAuley, 2018] [*Self-Attentive Sequential Recommendation*](https://arxiv.org/abs/1808.09781)                                                                            |   7.3946   |
| **BERT4Rec** | Bidirectional Encoder Representations from Transformers for Recommendation | [Sun et al., 2019] [*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*](https://arxiv.org/abs/1904.06690)                         |   8.4634   |

> BERT4Rec is trained with a masked-language-modeling objective rather than next-item prediction, so its loss is not directly comparable to the autoregressive models above.

---

### 5. Recall

Benchmarked on **Amazon Reviews — Beauty** (5-core, 2014).

| Model          | Full Name                                         | Paper                                                                                                                                                                                                                      |    Type     | Val Loss ↓ |
| :------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :--------: |
| **YouTubeDNN** | Deep Neural Network for YouTube Recommendations   | [Covington et al., 2016] [*Deep Neural Networks for YouTube Recommendations*](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)                                                              |   Neural    | **6.9905** |
| **Two-Tower**  | Two-Tower Neural Network                          | [Yang et al., 2020] [*Mixed Negative Sampling for Learning Two-Tower Neural Networks in Recommendations*](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/) |   Neural    |   7.4335   |
| **Item2Vec**   | Neural Item Embedding for Collaborative Filtering | [Barkan & Koenigstein, 2016] [*Item2Vec: Neural Item Embedding for Collaborative Filtering*](https://arxiv.org/abs/1603.04259)                                                                                             |   Neural    |   1.4852   |
| **ItemCF**     | Item-Based Collaborative Filtering                | [Sarwar et al., 2001] [*Item-Based Collaborative Filtering Recommendation Algorithms*](https://doi.org/10.1145/371920.372071)                                                                                              | Statistical |     —      |
| **Swing**      | Swing Item-to-Item Collaborative Filtering        | Alibaba, [*Swing Recommendation Algorithm*](https://www.alibabacloud.com/help/en/airec/what-is-pai-rec/user-guide/swing-algorithm-tools)                                                                                   | Statistical |     —      |

> ItemCF and Swing are non-parametric item-to-item methods with no training objective. Item2Vec optimizes a skip-gram objective, so its loss is not comparable to the softmax-based retrieval models above.

---

## Roadmap

- [ ] **TIGER** — Transformer Index for Generative Recommenders ([Rajput et al., 2023](https://arxiv.org/abs/2305.05065)): semantic ID quantization + generative retrieval
- [ ] **Retrieval metrics** — Recall@K / NDCG@K / Coverage@K across all recall models, enabling comparison between neural and statistical methods
- [ ] **Multi-seed evaluation** — variance bounds for multi-task models, where current differences fall within run-to-run noise
- [ ] **BST calibration** — investigate LogLoss anomaly under the current sequence configuration
## Contact
For questions, suggestions, or collaboration, please contact:
Email: guopan.jin@outlook.com
## Citation
If you find deepsklearn useful in your research or projects, please consider citing it:
```bibtex
@software{jin2026deepsklearn,
  author = {Jin, Guopan},
  title  = {{DeepSklearn}},
  year   = {2026},
  url    = {https://github.com/guopanjin/deepsklearn}
}
```

