# Awesome Resource-Efficient LLM Papers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
<div style="display: flex; align-items: center;">
  <div style="flex: 1;">
    WORK IN PROGRESS
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
    - [Data Efficiency](#data-efficiency)
  - [LLM Fine-Tuning](#llm-fine-tuning)
    - [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
    - [Full-Parameter Fine-Tuning](#full-parameter-fine-tuning)
  - [LLM Inference](#llm-inference)
    - [Model Compression](#model-compression)
    - [Dynamic Acceleration](#dynamic-acceleration)
  - [System Design](#system-design)
    - [Hardware Offloading](#hardware-offloading)
    - [Collaborative Inference](#collaborative-inference)
    - [Libraries](#libraries)
    - [Edge Devices](#edge-devices)
    - [Other Systems](#other-systems)
  - [LLM Resource Efficiency Leaderboards](#llm-resource-efficiency-leaderboards)


|  Date  |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     Transformers     |      Google      | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                   |   NeurIPS<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F204e3073870fae3d05bcbc2f6a8e263d9b72e776%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |

<!-------------------------------------------------------------------------------------->

 ## LLM Architecture Design
 ### Efficient Transformer Architecture
 - [Example](https://example.com/) - Description of an example paper.


 ### Non-transformer Architecture
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2017 | Mixture of Experts | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://openreview.net/forum?id=B1ckMDqlg) | ICLR | 
| 2022 | Mixture of Experts | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0998.html) | JMLR |
| 2022 | Mixture of Experts | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c.html) | ICML|
| 2022 | Mixture of Experts | [Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html) | NeurIPS |
| 2022 | Mixture of Experts | [Efficient Large Scale Language Modeling with Mixtures of Experts](https://aclanthology.org/2022.emnlp-main.804/) | EMNLP|
| 2023 | RNN LM | [RWKV: Reinventing RNNs for the Transformer Era](https://aclanthology.org/2023.findings-emnlp.936/) | EMNLP-Findings|
| 2023 | MLP | [Auto-Regressive Next-Token Predictors are Universal Learners](https://arxiv.org/abs/2309.06979) | ArXiv |
| 2023 | Convolutional LM| [Hyena Hierarchy: Towards Larger Convolutional Language models](https://arxiv.org/abs/2302.10866) | ICML|
| 2023 | Sub-quadratic Matrices based| [Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture](https://nips.cc/virtual/2023/poster/71105) | NeurIPS |
| 2023 | Selective State Space Model | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | ArXiv |

<!-------------------------------------------------------------------------------------->
 ## LLM Pre-Training
 ### Memory Efficiency
 #### Distributed Training
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2020 | Data Parallelism | [Zero: Memory optimizations toward training trillion parameter models](https://arxiv.org/abs/1910.02054) | IEEE SC20 | 
| 2021 | Data Parallelism | [FairScale:  A general purpose modular PyTorch library for high performance and large scale training](https://github.com/facebookresearch/fairscale) | JMLR | 
| 2023 | Data Parallelism | [Palm: Scaling language modeling with pathways](https://arxiv.org/abs/2204.02311) | Github | 
| 2018 | Model Parallelism | [Mesh-tensorflow: Deep learning for supercomputers](https://arxiv.org/abs/1811.02084) | NeurIPS |
| 2019 | Model Parallelism | [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) | NeurIPS |
| 2019 | Model Parallelism | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | Arxiv |
| 2019 | Model Parallelism | [PipeDream: generalized pipeline parallelism for DNN training](https://arxiv.org/abs/1806.03377) | SOSP |
| 2022 | Model Parallelism | [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) | OSDI |
| 2023 | Model Parallelism | [Bpipe: memory-balanced pipeline parallelism for training large language models](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf) | JMLR |

 #### Mixed precision training
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2017 | Mixed Precision Training| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | ICLR | 
| 2018 | Mixed Precision Training| [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) | ACL | 
| 2022 | Mixed Precision Training| [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) | Arxiv | 


 ### Data Efficiency
 - [Example](https://example.com/) - Description of an example paper.

<!-------------------------------------------------------------------------------------->
 ## LLM Fine-Tuning
 ### Parameter-Efficient Fine-Tuning
 |  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2019 | Masking-based fine-tuning | [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/abs/1911.03437) | ACL | 
| 2021 | Masking-based fine-tuning | [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199) | ACL |
| 2021 | Masking-based fine-tuning | [Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning](https://arxiv.org/abs/2109.05687) | EMNLP |
| 2021 | Masking-based fine-tuning | [Unlearning Bias in Language Models by Partitioning Gradients](https://aclanthology.org/2023.findings-acl.375.pdf) | ACL |
| 2022 | Masking-based fine-tuning | [Fine-Tuning Pre-Trained Language Models Effectively by Optimizing Subnetworks Adaptively](https://openreview.net/pdf?id=-r6-WNKfyhW) | NeurIPS |

 ### Full-Parameter Fine-Tuning
 - [Example](https://example.com/) - Description of an example paper.

<!-------------------------------------------------------------------------------------->
 ## LLM Inference

 ### Model Compression

 #### Pruning
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
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
| 2021 | Activation Quantization | [Ac-gc: Lossy activation compression with guaranteed convergence](https://proceedings.neurips.cc/paper/2021/hash/e655c7716a4b3ea67f48c6322fc42ed6-Abstract.html) | NeurIPS |
| 2022 | Fixed-point Quantization | [Boost Vision Transformer with GPU-Friendly Sparsity and Quantization](https://arxiv.org/abs/2305.10727) | ACL |

 ### Dynamic Acceleration
 
 #### Input Pruning
|  Date  |       Keywords     | Paper    | Venue |
| :---------: | :------------: | :-----------------------------------------:| :---------: |
| 2021 | Score-based Token Removal | [Efficient sparse attention architecture with cascade token and head pruning](https://arxiv.org/pdf/2012.09852) | HPCA | 
| 2022 | Score-based Token Removal | [Learned Token Pruning for Transformers](https://dl.acm.org/doi/abs/10.1145/3534678.3539260) | KDD | 
| 2023 | Score-based Token Removal | [Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference](https://arxiv.org/abs/2306.14393) | KDD | 
| 2021 | Learning-based Token Removal | [TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference](https://arxiv.org/abs/2105.11618) | NAACL | 
| 2022 | Learning-based Token Removal | [Transkimmer: Transformer Learns to Layer-wise Skim](https://arxiv.org/abs/2205.07324) | ACL | 
| 2023 | Learning-based Token Removal | [PuMer: Pruning and Merging Tokens for Efficient Vision Language Models](https://arxiv.org/abs/2305.17530) | ACL | 
| 2023 | Learning-based Token Removal | [Infor-Coef: Information Bottleneck-based Dynamic Token Downsampling for Compact and Efficient language model](https://arxiv.org/abs/2305.12458) | arXiv | 
| 2023 | Learning-based Token Removal | [SmartTrim: Adaptive Tokens and Parameters Pruning for Efficient Vision-Language Models](https://arxiv.org/abs/2305.15033) | arXiv | 

<!-------------------------------------------------------------------------------------->
 ## System Design

 ### Hardware Offloading
  - [Example](https://example.com/) - Description of an example paper.

 ### Collaborative Inference
  - [Example](https://example.com/) - Description of an example paper.

 ### Libraries
  - [Example](https://example.com/) - Description of an example paper.

 ### Edge Devices
  - [Example](https://example.com/) - Description of an example paper.

 ### Other Systems
  - [Example](https://example.com/) - Description of an example paper.

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
  > - [CodeCarbon](https://codecarbon.io/) - a lightweight Python-compatible package that quantifies the carbon dioxide emissions generated by computing resources and provides methods for reducing the environmental impact.
  > - [Carbontracker](https://github.com/lfwa/carbontracker) - 
  > - [experiment-impact-tracker](https://github.com/Breakend/experiment-impact-tracker)

<!-- tools for predicting the energy usage and carbon footprint before training**. -->
  > You might also find the following helpful for predicting the energy usage and carbon footprint before actual training or 
  > - [ML CO2 Impact](https://mlco2.github.io/impact/) - a web-based tool that estimates the carbon emission of a model by estimating the electricity consumption of the training procedure.
  > - [LLMCarbon](https://github.com/SotaroKaneda/MLCarbon) - 

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

   - [Example](https://example.com/) - Description of an example paper.

<!-------------------------------------------------------------------------------------->
 ## Reference
If you find this paper list useful in your research, please consider citing:

    @article{bai2024beyond,
      title={Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models},
      author={Bai, Guangji and Chai, Zheng and Ling, Chen and Wang, Shiyu and Lu, Jiaying and Zhang, Nan and Shi, Tingwei and Yu, Ziyang and Zhu, Mengdan and Zhang, Yifei and others},
      journal={arXiv preprint arXiv:2401.00625},
      year={2024}
    }
