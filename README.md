# Awesome Resource-Efficient LLM Papers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
<div style="display: flex; align-items: center;">
  <div style="flex: 1;">
    A curated list of high-quality papers on resource-efficient LLMs. 
  </div>
  <div>
    <img src="media/clean_energy.gif" alt="Clean Energy GIF" width="80" />
  </div>
</div>

This is the GitHub repo for our survey paper [Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models](https://arxiv.org/pdf/2401.00625).

 ## Table of Contents
  - [LLM Architecture Design](#llm-architecture-design)
    - [Efficient Transformer Architecture](#efficient-transformer-architecture)
    - [Non-transformer Architecture](#non-transformer-architecture)
  - [LLM Pre-Training](#llm-pre-training)
    - [Memory Efficiency](#memory-efficiency)
      - [Distributed Training](#distributed-training)
      - [Mixed Precision Training](#mixed-precision-training)
    - [Data Efficiency](#data-efficiency)
      - [Importance Sampling](#importance-sampling)
      - [Data Augmentation](#data-augmentation)
      - [Training Objective](#training-objective)
  - [LLM Fine-Tuning](#llm-fine-tuning)
    - [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
    - [Full-Parameter Fine-Tuning](#full-parameter-fine-tuning)
  - [LLM Inference](#llm-inference)
    - [Model Compression](#model-compression)
      - [Pruning](#pruning)
      - [Quantization](#quantization)
    - [Dynamic Acceleration](#dynamic-acceleration)
      - [Input Pruning](#input-pruning)
  - [System Design](#system-design)
    - [Deployment Optimization](#deployment-optimization)
    - [Support Infrastructure](#support-infrastructure)
    - [Other Systems](#other-systems)
  - [Resource-Efficiency Evaluation Metrics & Benchmarks](#resource-efficiency-evaluation-metrics--benchmarks)
    - [🧮 Computation Metrics](#🧮-computation-metrics)
    - [💾 Memory Metrics](#💾-memory-metrics)
    - [⚡️ Energy Metrics](#⚡️-energy-metrics)
    - [💵 Financial Cost Metric](#💵-financial-cost-metric)
    - [📨 Network Communication Metric](#📨-network-communication-metric)
    - [💡 Other Metrics](#💡-other-metrics)
    - [Benchmarks](#benchmarks)
  - [Reference](#reference)

<!-------------------------------------------------------------------------------------->

 ## LLM Architecture Design
 ### Efficient Transformer Architecture
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2024 | Hardware optimization | [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | ICLR |
| 2023 | Hardware optimization | [Flashattention: Fast and memory-efficient exact attention with io-awareness](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) | NeurIPS |
| 2023 | Approximate attention | [KDEformer: Accelerating Transformers via Kernel Density Estimation](https://arxiv.org/abs/2302.02451) | ICML|
| 2023 | Approximate attention | [Mega: Moving Average Equipped Gated Attention ](https://openreview.net/forum?id=qNLe3iq2El) | ICLR|
| 2022 | Hardware optimization | [xFormers - Toolbox to Accelerate Research on Transformers](https://github.com/facebookresearch/xformers) | GitHub|
| 2021 | Approximate attention | [Efficient attention: Attention with linear complexities](https://arxiv.org/abs/1812.01243) | WACV |
| 2021 | Approximate attention | [An Attention Free Transformer](https://arxiv.org/pdf/2105.14103.pdf)| ArXiv |
| 2021 | Approximate attention | [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682) | ArXiv |
| 2021 | Hardware optimization | [LightSeq: A High Performance Inference Library for Transformers](https://aclanthology.org/2021.naacl-industry.15/)| NAACL|
| 2021 | Hardware optimization | [FasterTransformer: A Faster Transformer Framework](https://github.com/NVIDIA/FasterTransformer)| GitHub|
| 2020 | Approximate attention | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)| ICML |
| 2019 | Approximate attention | [Reformer: The efficient transformer](https://openreview.net/forum?id=rkgNKkHtvB)  | ICLR |

 ### Non-transformer Architecture
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | RNN LM | [RWKV: Reinventing RNNs for the Transformer Era](https://aclanthology.org/2023.findings-emnlp.936/) | EMNLP-Findings|
| 2023 | MLP | [Auto-Regressive Next-Token Predictors are Universal Learners](https://arxiv.org/abs/2309.06979) | ArXiv |
| 2023 | Convolutional LM| [Hyena Hierarchy: Towards Larger Convolutional Language models](https://arxiv.org/abs/2302.10866) | ICML|
| 2023 | Sub-quadratic Matrices based| [Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture](https://nips.cc/virtual/2023/poster/71105) | NeurIPS |
| 2023 | Selective State Space Model | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | ArXiv |
| 2022 | Mixture of Experts | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0998.html) | JMLR |
| 2022 | Mixture of Experts | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c.html) | ICML|
| 2022 | Mixture of Experts | [Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html) | NeurIPS |
| 2022 | Mixture of Experts | [Efficient Large Scale Language Modeling with Mixtures of Experts](https://aclanthology.org/2022.emnlp-main.804/) | EMNLP|
| 2017 | Mixture of Experts | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://openreview.net/forum?id=B1ckMDqlg) | ICLR | 

<!-------------------------------------------------------------------------------------->
 ## LLM Pre-Training
 ### Memory Efficiency
 #### Distributed Training
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Data Parallelism | [Palm: Scaling language modeling with pathways](https://arxiv.org/abs/2204.02311) | Github | 
| 2023 | Model Parallelism | [Bpipe: memory-balanced pipeline parallelism for training large language models](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf) | JMLR |
| 2022 | Model Parallelism | [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) | OSDI |
| 2021 | Data Parallelism | [FairScale:  A general purpose modular PyTorch library for high performance and large scale training](https://github.com/facebookresearch/fairscale) | JMLR|
| 2020 | Data Parallelism | [Zero: Memory optimizations toward training trillion parameter models](https://arxiv.org/abs/1910.02054) | IEEE SC20 | 
| 2019 | Model Parallelism | [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) | NeurIPS |
| 2019 | Model Parallelism | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | Arxiv |
| 2019 | Model Parallelism | [PipeDream: generalized pipeline parallelism for DNN training](https://arxiv.org/abs/1806.03377) | SOSP |
| 2018 | Model Parallelism | [Mesh-tensorflow: Deep learning for supercomputers](https://arxiv.org/abs/1811.02084) | NeurIPS |

 #### Mixed precision training
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2022 | Mixed Precision Training| [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) | Arxiv |
| 2018 | Mixed Precision Training| [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) | ACL | 
| 2017 | Mixed Precision Training| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | ICLR | 

 ### Data Efficiency
 #### Importance Sampling
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Survey on importance sampling | [A Survey on Efficient Training of Transformers](https://arxiv.org/abs/2302.01107) | IJCAI | 
| 2023 | Importance sampling | [Data-Juicer: A One-Stop Data Processing System for Large Language Models](https://arxiv.org/abs/2309.02033) | Arxiv | 
| 2023 | Importance sampling | [INGENIOUS: Using Informative Data Subsets for Efficient Pre-Training of Language Models](https://aclanthology.org/2023.findings-emnlp.445/) | EMNLP | 
| 2023 | Importance sampling | [Machine Learning Force Fields with Data Cost Aware Training](https://arxiv.org/abs/2306.03109) | ICML | 
| 2022 | Importance sampling | [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486) | NeurIPS |
| 2021 | Importance sampling | [Deep Learning on a Data Diet: Finding Important Examples Early in Training](https://arxiv.org/abs/2107.07075) | NeurIPS | 
| 2018 | Importance sampling | [Training Deep Models Faster with Robust, Approximate Importance Sampling](https://proceedings.neurips.cc/paper/2018/hash/967990de5b3eac7b87d49a13c6834978-Abstract.html) | NeurIPS | 
| 2018 | Importance sampling | [Not All Samples Are Created Equal: Deep Learning with Importance Sampling](http://proceedings.mlr.press/v80/katharopoulos18a.html) | ICML | 

 #### Data Augmentation
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Data augmentation | [MixGen: A New Multi-Modal Data Augmentation](https://openaccess.thecvf.com/content/WACV2023W/Pretrain/html/Hao_MixGen_A_New_Multi-Modal_Data_Augmentation_WACVW_2023_paper.html) | WACV | 
| 2023 | Data augmentation | [Augmentation-Aware Self-Supervision for Data-Efficient GAN Training](https://arxiv.org/abs/2205.15677) | NeurIPS | 
| 2023 | Data augmentation | [Improving End-to-End Speech Processing by Efficient Text Data Utilization with Latent Synthesis](https://aclanthology.org/2023.findings-emnlp.327/) | EMNLP | 
| 2023 | Data augmentation | [FaMeSumm: Investigating and Improving Faithfulness of Medical Summarization](https://aclanthology.org/2023.emnlp-main.673/) | EMNLP | 

 #### Training Objective
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Training objective | [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.10169) | Arxiv | 
| 2023 | Training objective | [Efficient Data Learning for Open Information Extraction with Pre-trained Language Models](https://aclanthology.org/2023.findings-emnlp.869/) | EMNLP |
| 2023 | Masked language-image modeling | [Scaling Language-Image Pre-training via Masking](https://arxiv.org/abs/2212.00794) | CVPR |
| 2022 | Masked image modeling | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | CVPR | 
| 2019 | Masked language modeling | [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450) | ICML | 

<!-------------------------------------------------------------------------------------->
 ## LLM Fine-Tuning
 ### Parameter-Efficient Fine-Tuning
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2022 | Masking-based fine-tuning | [Fine-Tuning Pre-Trained Language Models Effectively by Optimizing Subnetworks Adaptively](https://openreview.net/pdf?id=-r6-WNKfyhW) | NeurIPS |
| 2021 | Masking-based fine-tuning | [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199) | ACL |
| 2021 | Masking-based fine-tuning | [Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning](https://arxiv.org/abs/2109.05687) | EMNLP |
| 2021 | Masking-based fine-tuning | [Unlearning Bias in Language Models by Partitioning Gradients](https://aclanthology.org/2023.findings-acl.375.pdf) | ACL |
| 2019 | Masking-based fine-tuning | [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/abs/1911.03437) | ACL |

 ### Full-Parameter Fine-Tuning
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Comparative study betweeen full-parameter and LoRA-base fine-tuning | [A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Instruction Following Large Language Model](https://arxiv.org/abs/2304.08109) | Arxiv | 
| 2023 | Comparative study betweeen full-parameter and parameter-efficient fine-tuning | [Comparison between parameter-efficient techniques and full fine-tuning: A case study on multilingual news article classification](https://arxiv.org/abs/2308.07282) | Arxiv |
| 2023 | Full-parameter fine-tuning with limited resources | [Full Parameter Fine-tuning for Large Language Models with Limited Resources](https://arxiv.org/abs/2306.09782) | Arxiv |
| 2023 | Memory-efficient fine-tuning | [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333) | NeurIPS |
| 2023 | Full-parameter fine-tuning for medicine applications | [PMC-LLaMA: Towards Building Open-source Language Models for Medicine](https://arxiv.org/abs/2304.14454) | Arxiv |
| 2022 | Drawback of full-parameter fine-tuning | [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://openreview.net/forum?id=UYneFzXSJWh) | ICLR |

<!-------------------------------------------------------------------------------------->
 ## LLM Inference

 ### Model Compression

 #### Pruning
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2024 | Structured Pruning | [Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models](https://arxiv.org/pdf/2405.20541) | Arxiv |
| 2024 | Structured Pruning | [BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation](https://arxiv.org/pdf/2402.16880) | Arxiv |
| 2024 | Structured Pruning | [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/pdf/2403.03853) | Arxiv |
| 2024 | Structured Pruning | [NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models](https://arxiv.org/pdf/2402.09773) | Arxiv |
| 2024 | Structured Pruning | [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](https://arxiv.org/pdf/2401.15024) | ICLR |
| 2024 | Unstructured Pruning | [Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs](https://arxiv.org/pdf/2310.08915) | ICLR |
| 2024 | Structured Pruning | [Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models](https://openreview.net/pdf?id=Tr0lPx9woF) | ICLR |
| 2023 | Unstructured Pruning | [One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models](https://arxiv.org/pdf/2310.09499) | Arxiv |
| 2023 | Unstructured Pruning | [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774) | ICML | 
| 2023 | Unstructured Pruning | [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695) | ICLR | 
| 2023 | Unstructured Pruning | [AccelTran: A Sparsity-Aware Accelerator for Dynamic Inference With Transformers](https://ieeexplore.ieee.org/abstract/document/10120981) | TCAD | 
| 2023 | Structured Pruning | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | NeurIPS | 
| 2023 | Structured Pruning | [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222) | ICML | 
| 2023 | Structured Pruning | [Structured Pruning for Efficient Generative Pre-trained Language Models](https://aclanthology.org/2023.findings-acl.692) | ACL | 
| 2023 | Structured Pruning | [ZipLM: Inference-Aware Structured Pruning of Language Models](https://arxiv.org/abs/2302.04089) | NeurIPS | 
| 2023 | Contextual Pruning | [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://proceedings.mlr.press/v202/liu23am.html) | ICML | 

 #### Quantization
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Weight Quantization | [Flexround: Learnable rounding based on element-wise division for post-training quantization](https://arxiv.org/abs/2306.00317) | ICML | 
| 2023 | Weight Quantization | [Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling](https://arxiv.org/abs/2304.09145) | EMNLP | 
| 2023 | Weight Quantization | [OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://arxiv.org/abs/2306.02272) | AAAI | 
| 2023 | Weight Quantization | [Gptq: Accurate posttraining quantization for generative pre-trained transformers](https://arxiv.org/abs/2210.17323) | ICLR |
| 2023 | Weight Quantization | [Dynamic Stashing Quantization for Efficient Transformer Training](https://arxiv.org/abs/2303.05295) | EMNLP |
| 2023 | Weight Quantization | [Quantization-aware and tensor-compressed training of transformers for natural language understanding](https://arxiv.org/abs/2306.01076) | Interspeech |
| 2023 | Weight Quantization | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | NeurIPS |
| 2023 | Weight Quantization | [Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013) | NeurIPS |
| 2023 | Weight Quantization | [Prequant: A task-agnostic quantization approach for pre-trained language models](https://arxiv.org/abs/2306.00014) | ACL |
| 2023 | Weight Quantization | [Olive: Accelerating large language models via hardware-friendly outliervictim pair quantization](https://arxiv.org/abs/2304.07493) | ISCA |
| 2023 | Weight Quantization | [Awq: Activationaware weight quantization for llm compression and acceleration](https://arxiv.org/abs/2306.00978) | arXiv | 
| 2023 | Weight Quantization | [Spqr: A sparsequantized representation for near-lossless llm weight compression](https://arxiv.org/abs/2306.03078) | arXiv | 
| 2023 | Weight Quantization | [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629) | arXiv | 
| 2023 | Weight Quantization | [LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/abs/2305.17888) | arXiv |
| 2022 | Activation Quantization | [Gact: Activation compressed training for generic network architectures](https://arxiv.org/abs/2206.11357) | ICML |
| 2022 | Fixed-point Quantization | [Boost Vision Transformer with GPU-Friendly Sparsity and Quantization](https://arxiv.org/abs/2305.10727) | ACL |
| 2021 | Activation Quantization | [Ac-gc: Lossy activation compression with guaranteed convergence](https://proceedings.neurips.cc/paper/2021/hash/e655c7716a4b3ea67f48c6322fc42ed6-Abstract.html) | NeurIPS |

 ### Dynamic Acceleration
 
 #### Input Pruning
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Score-based Token Removal | [Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference](https://arxiv.org/abs/2306.14393) | KDD | 
| 2023 | Learning-based Token Removal | [PuMer: Pruning and Merging Tokens for Efficient Vision Language Models](https://arxiv.org/abs/2305.17530) | ACL | 
| 2023 | Learning-based Token Removal | [Infor-Coef: Information Bottleneck-based Dynamic Token Downsampling for Compact and Efficient language model](https://arxiv.org/abs/2305.12458) | arXiv | 
| 2023 | Learning-based Token Removal | [SmartTrim: Adaptive Tokens and Parameters Pruning for Efficient Vision-Language Models](https://arxiv.org/abs/2305.15033) | arXiv |
| 2022 | Learning-based Token Removal | [Transkimmer: Transformer Learns to Layer-wise Skim](https://arxiv.org/abs/2205.07324) | ACL | 
| 2022 | Score-based Token Removal | [Learned Token Pruning for Transformers](https://dl.acm.org/doi/abs/10.1145/3534678.3539260) | KDD | 
| 2021 | Learning-based Token Removal | [TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference](https://arxiv.org/abs/2105.11618) | NAACL | 
| 2021 | Score-based Token Removal | [Efficient sparse attention architecture with cascade token and head pruning](https://arxiv.org/pdf/2012.09852) | HPCA | 

<!-------------------------------------------------------------------------------------->
 ## System Design

 ### Deployment optimization
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Hardware offloading | [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865) | PMLR | 
| 2023 | Hardware offloading | [Fast distributed inference serving for large language models](https://arxiv.org/abs/2305.05920) | arXiv | 
| 2022 | Collaborative inference | [Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188) | arXiv | 
| 2022 | Hardware offloading | [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032) | IEEE SC22 | 

 ### Support Infrastructure
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Edge devices | [Training Large-Vocabulary Neural Language Models by Private Federated Learning for Resource-Constrained Devices](https://arxiv.org/abs/2207.08988) | ICASSP |
| 2023 | Edge devices | [Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly](https://arxiv.org/abs/2310.03150) | arXiv |
| 2023 | Libraries | [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883) | ICPP | 
| 2023 | Libraries | [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745) | ACL | 
| 2023 | Edge devices | [Large Language Models Empowered Autonomous Edge AI for Connected Intelligence](https://arxiv.org/abs/2307.02779) | arXiv |
| 2022 | Libraries | [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032) | IEEE SC22 | 
| 2022 | Libraries | [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) | OSDI | 
| 2022 | Edge devices | [EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation](https://arxiv.org/abs/2202.07959) | arXiv |
| 2022 | Edge devices | [ProFormer: Towards On-Device LSH Projection-Based Transformers](https://arxiv.org/abs/2004.05801) | ACL |
| 2021 | Edge devices | [Generate More Features with Cheap Operations for BERT](https://aclanthology.org/2021.acl-long.509.pdf) | ACL |
| 2021 | Edge devices | [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) | SustaiNLP |
| 2020 | Edge devices | [Lite Transformer with Long-Short Range Attention](https://arxiv.org/abs/2004.11886) | arXiv |
| 2019 | Libraries | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | IEEE SC22 | 
| 2018 | Libraries | [Mesh-TensorFlow: Deep Learning for Supercomputers](https://arxiv.org/abs/1811.02084) | NeurIPS | 

 ### Other Systems
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2023 | Other Systems | [Tabi: An Efficient Multi-Level Inference System for Large Language Models](https://dl.acm.org/doi/abs/10.1145/3552326.3587438) | EuroSys | 
| 2023 | Other Systems | [Near-Duplicate Sequence Search at Scale for Large Language Model Memorization Evaluation](https://dl.acm.org/doi/abs/10.1145/3589324) | PACMMOD | 


<!-------------------------------------------------------------------------------------->
 ## Resource-Efficiency Evaluation Metrics \& Benchmarks
 ### 🧮 Computation Metrics

|  Metric                           |       Description                                                |    Example Usage                                                                                | 
| :-------------------------------: | :--------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| FLOPs (Floating-point operations) | the number of arithmetic operations on floating-point numbers    | [\[FLOPs\]](https://arxiv.org/pdf/2309.13192.pdf) |
| Training Time                     | the total duration required for training, typically measured in wall-clock minutes, hours, or days  |[\[minutes, days\]](https://arxiv.org/pdf/2309.13192.pdf)<br>[\[hours\]](https://www.jmlr.org/papers/volume24/23-0069/23-0069.pdf)|
| Inference Time/Latency            | the average time required generate an output after receiving an input, typically measured in wall-clock time or CPU/GPU/TPU clock time in milliseconds or seconds  |[\[end-to-end latency in seconds\]](https://arxiv.org/pdf/2309.06180.pdf)<br>[\[next token generation latency in milliseconds\]](https://arxiv.org/pdf/2311.00502.pdf)|
| Throughput                        | the rate of output tokens generation or tasks completion, typically measured in tokens per second (TPS) or queries per second (QPS) |[\[tokens/s\]](https://arxiv.org/pdf/2209.01188.pdf)<br>[\[queries/s\]](https://arxiv.org/pdf/2210.16773.pdf)|
| Speed-Up Ratio                    | the improvement in inference speed compared to a baseline model |[\[inference time speed-up\]](https://aclanthology.org/2021.naacl-industry.15.pdf)<br>[\[throughput speed-up\]](https://github.com/NVIDIA/FasterTransformer)|

### 💾 Memory Metrics

|  Metric              |       Description                                              |    Example Usage                                                                                | 
| :------------------: | :---------------------------------------------------------:    | :---------------------------------------------------------------------------------------------: |
| Number of Parameters | the number of adjustable variables in the LLM’s neural network | [\[number of parameters\]](https://arxiv.org/pdf/1706.03762.pdf)|
| Model Size           | the storage space required for storing the entire model        | [\[peak memory usage in GB\]](https://arxiv.org/pdf/2302.02451.pdf)|

### ⚡️ Energy Metrics

|  Metric              |       Description                                              |    Example Usage                                                                                | 
| :------------------: | :---------------------------------------------------------:    | :---------------------------------------------------------------------------------------------: |
| Energy Consumption   | the electrical power used during the LLM’s lifecycle           | [\[kWh\]](https://arxiv.org/pdf/1906.02243.pdf)|
| Carbon Emission      | the greenhouse gas emissions associated with the model’s energy usage |[\[kgCO2eq\]](https://jmlr.org/papers/volume21/20-312/20-312.pdf)|

<!-- software packages designed for real-time tracking of energy consumption and carbon emissions**. -->
  > The following are available software packages designed for real-time tracking of energy consumption and carbon emission.
  > - [CodeCarbon](https://codecarbon.io/)
  > - [Carbontracker](https://github.com/lfwa/carbontracker)
  > - [experiment-impact-tracker](https://github.com/Breakend/experiment-impact-tracker)

<!-- tools for predicting the energy usage and carbon footprint before training**. -->
  > You might also find the following helpful for predicting the energy usage and carbon footprint before actual training or 
  > - [ML CO2 Impact](https://mlco2.github.io/impact/)
  > - [LLMCarbon](https://github.com/SotaroKaneda/MLCarbon)

### 💵 Financial Cost Metric
|  Metric               |       Description                                              |    Example Usage                                                                                | 
| :------------------:  | :---------------------------------------------------------:    | :---------------------------------------------------------------------------------------------: |
| Dollars per parameter | the total cost of training (or running) the LLM by the number of parameters | |

### 📨 Network Communication Metric
|  Metric               |       Description                                              |    Example Usage                                                                                | 
| :------------------:  | :---------------------------------------------------------:    | :---------------------------------------------------------------------------------------------: |
| Communication Volume  | the total amount of data transmitted across the network during a specific LLM execution or training run | [\[communication volume in TB\]](https://arxiv.org/pdf/2310.06003.pdf)|

### 💡 Other Metrics
|  Metric               |       Description                                              |    Example Usage                                                                                | 
| :------------------:  | :---------------------------------------------------------:    | :---------------------------------------------------------------------------------------------: |
| Compression Ratio     |  the reduction in size of the compressed model compared to the original model | [\[compress rate\]](https://arxiv.org/pdf/1510.00149.pdf)<br>[\[percentage of weights remaining\]](https://arxiv.org/pdf/2306.11222.pdf)|
| Loyalty/Fidelity      |  the resemblance between the teacher and student models in terms of both predictions consistency and predicted probability distributions alignment | [\[loyalty\]](https://arxiv.org/pdf/2109.03228.pdf)<br>[\[fidelity\]](https://arxiv.org/pdf/2106.05945.pdf)|
| Robustness            |  the resistance to adversarial attacks, where slight input modifications can potentially manipulate the model's output | [\[after-attack accuracy, query number\]](https://arxiv.org/pdf/2109.03228.pdf)|
| Pareto Optimality     |  the optimal trade-offs between various competing factors | [\[Pareto frontier (cost and accuracy)\]](https://arxiv.org/pdf/2212.01340.pdf)<br>[\[Pareto frontier (performance and FLOPs)\]](https://arxiv.org/pdf/2110.07038.pdf)|
### Benchmarks
|  Benchmark               |       Description                                              |    Paper                                                                                | 
| :------------------:  | :---------------------------------------------------------:    | :---------------------------------------------------------------------------------------------: |
| General NLP Benchmarks |  an extensive collection of general NLP benchmarks such as [GLUE](https://arxiv.org/pdf/1804.07461.pdf), [SuperGLUE](https://arxiv.org/pdf/1905.00537.pdf), [WMT](https://aclanthology.org/W16-2301.pdf), and [SQuAD](https://arxiv.org/pdf/1606.05250.pdf), etc. | [A Comprehensive Overview of Large Language Models](https://arxiv.org/pdf/2307.06435.pdf)|
| Dynaboard      |  an open-source platform for evaluating NLP models in the cloud, offering real-time interaction and a holistic assessment of model quality with customizable Dynascore | [Dynaboard: An Evaluation-As-A-Service Platform for Holistic Next-Generation Benchmarking](https://proceedings.neurips.cc/paper_files/paper/2021/file/55b1927fdafef39c48e5b73b5d61ea60-Paper.pdf)|
| EfficientQA            |  an open-domain Question Answering (QA) challenge at [NeurIPS 2020](https://efficientqa.github.io/) that focuses on building accurate, memory-efficient QA systems | [NeurIPS 2020 EfficientQA Competition: Systems, Analyses and Lessons Learned](https://proceedings.mlr.press/v133/min21a/min21a.pdf)|
| [SustaiNLP 2020](https://sites.google.com/view/sustainlp2020) Shared Task     |  a challenge for development of energy-efficient NLP models by assessing their performance across eight NLU tasks using SuperGLUE metrics and evaluating their energy consumption during inference | [Overview of the SustaiNLP 2020 Shared Task](https://aclanthology.org/2020.sustainlp-1.24.pdf)|
| ELUE (Efficient Language Understanding Evaluation)     |  a benchmark platform for evaluating NLP model efficiency across various tasks, offering online metrics and requiring only a Python model definition file for submission | [Towards Efficient NLP: A Standard Evaluation and A Strong Baseline](https://arxiv.org/pdf/2110.07038.pdf)|
| VLUE (Vision-Language Understanding Evaluation)     |  a comprehensive benchmark for assessing vision-language models across multiple tasks, offering an online platform for evaluation and comparison | [VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models](https://arxiv.org/pdf/2205.15237.pdf)|
| Long Range Arena (LAG)     |  a benchmark suite evaluating efficient Transformer models on long-context tasks, spanning diverse modalities and reasoning types while allowing evaluations under controlled resource constraints, highlighting real-world efficiency | [Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/pdf/2011.04006.pdf)|
| Efficiency-aware MS MARCO  |  an enhanced [MS MARCO](https://arxiv.org/pdf/1611.09268.pdf) information retrieval benchmark that integrates efficiency metrics like per-query latency and cost alongside accuracy, facilitating a comprehensive evaluation of IR systems | [Moving Beyond Downstream Task Accuracy for Information Retrieval Benchmarking](https://arxiv.org/pdf/2212.01340.pdf)|

<!-------------------------------------------------------------------------------------->
 ## Reference
If you find this paper list useful in your research, please consider citing:

    @article{bai2024beyond,
      title={Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models},
      author={Bai, Guangji and Chai, Zheng and Ling, Chen and Wang, Shiyu and Lu, Jiaying and Zhang, Nan and Shi, Tingwei and Yu, Ziyang and Zhu, Mengdan and Zhang, Yifei and others},
      journal={arXiv preprint arXiv:2401.00625},
      year={2024}
    }
