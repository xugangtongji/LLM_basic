# AI 大模型理论与应用实践学习大纲

> 本大纲覆盖从数学基础到工程落地、再到前沿方向的完整学习路径，建议按章节循序渐进，也可根据需求跳跃式学习。

---

## 第一部分 数学与编程基础

### 1.1 数学基础
- 1.1.1 线性代数：向量、矩阵运算、特征值与特征向量、SVD、矩阵求导
- 1.1.2 概率与统计：随机变量、常见分布、贝叶斯定理、MLE / MAP、采样方法
- 1.1.3 微积分与最优化：偏导与梯度、链式法则、凸优化基础、拉格朗日乘子
- 1.1.4 信息论：信息熵、交叉熵、KL 散度、JS 散度、互信息、困惑度（Perplexity）
- 1.1.5 图论与离散数学（可选，用于图神经网络与知识图谱）

### 1.2 编程与工程基础
- 1.2.1 Python 核心：面向对象、装饰器、异步编程、类型提示
- 1.2.2 科学计算栈：NumPy、Pandas、Matplotlib、SciPy
- 1.2.3 PyTorch 基础：Tensor、Autograd、nn.Module、DataLoader、分布式 API
- 1.2.4 可选框架：JAX / Flax、TensorFlow / Keras
- 1.2.5 Linux 命令行、Git 版本控制、Docker 容器化
- 1.2.6 GPU 与 CUDA 基础：显存模型、计算流、Tensor Core、性能剖析

---

## 第二部分 机器学习与深度学习基础

### 2.1 机器学习基础
- 2.1.1 学习范式：监督学习、无监督学习、半监督、自监督、强化学习
- 2.1.2 核心概念：偏差-方差、过拟合、正则化、泛化误差
- 2.1.3 经典算法：线性/逻辑回归、SVM、决策树、随机森林、GBDT / XGBoost
- 2.1.4 聚类与降维：K-means、层次聚类、PCA、t-SNE、UMAP
- 2.1.5 评估方法：交叉验证、混淆矩阵、ROC / AUC、精准率 / 召回率 / F1

### 2.2 深度学习基础
- 2.2.1 神经网络原理：前向传播、反向传播、梯度消失 / 爆炸
- 2.2.2 激活函数：Sigmoid、Tanh、ReLU、GELU、SwiGLU
- 2.2.3 参数初始化：Xavier、He 初始化
- 2.2.4 规范化：BatchNorm、LayerNorm、RMSNorm、GroupNorm
- 2.2.5 经典网络：MLP、CNN、RNN、LSTM、GRU
- 2.2.6 优化器：SGD、Momentum、RMSProp、Adam、AdamW、Lion、Sophia
- 2.2.7 学习率调度：Warmup、Cosine、线性衰减、One-Cycle
- 2.2.8 正则化技巧：Dropout、权重衰减、标签平滑、数据增强
- 2.2.9 残差连接与注意力机制雏形

---

## 第三部分 NLP 基础

### 3.1 传统 NLP
- 3.1.1 预处理：分词、词形还原、停用词
- 3.1.2 文本表示：词袋模型、TF-IDF、BM25
- 3.1.3 语言模型：N-gram、平滑技术、困惑度
- 3.1.4 序列标注：HMM、CRF、POS / NER

### 3.2 词向量与表示学习
- 3.2.1 Word2Vec：CBOW、Skip-gram、负采样
- 3.2.2 GloVe、FastText
- 3.2.3 上下文相关表示：ELMo、CoVe

### 3.3 序列到序列模型
- 3.3.1 Encoder-Decoder 框架
- 3.3.2 注意力机制起源：Bahdanau Attention、Luong Attention
- 3.3.3 机器翻译与文本摘要的经典应用

---

## 第四部分 Transformer 架构

### 4.1 Transformer 原理
- 4.1.1 Self-Attention 数学推导
- 4.1.2 Multi-Head Attention
- 4.1.3 位置编码：绝对位置、相对位置、RoPE、ALiBi、NoPE
- 4.1.4 Encoder、Decoder、Encoder-Decoder 三种结构对比
- 4.1.5 Pre-LN vs Post-LN、残差连接
- 4.1.6 Feed-Forward Network、激活函数演进（ReLU → GeLU → SwiGLU）
- 4.1.7 模型参数量与计算量估算（FLOPs / 参数 / 显存）

### 4.2 注意力机制变体与优化
- 4.2.1 Causal Mask、Padding Mask
- 4.2.2 稀疏注意力：Longformer、BigBird、Sparse Transformer
- 4.2.3 Flash Attention v1 / v2 / v3
- 4.2.4 Multi-Query Attention（MQA）与 Grouped-Query Attention（GQA）
- 4.2.5 线性注意力与核化注意力
- 4.2.6 状态空间模型：S4、Mamba、Mamba-2、混合架构（Jamba）

### 4.3 经典模型家族
- 4.3.1 Encoder 系：BERT、RoBERTa、ALBERT、DeBERTa、ELECTRA
- 4.3.2 Decoder 系：GPT-1/2/3/3.5/4/4o/5、Claude 系列、LLaMA 1/2/3、Mistral、Qwen、DeepSeek
- 4.3.3 Encoder-Decoder 系：T5、BART、UL2
- 4.3.4 开源与闭源模型全景图

---

## 第五部分 大模型预训练

### 5.1 数据工程
- 5.1.1 数据来源：Common Crawl、代码、书籍、论文、对话、合成数据
- 5.1.2 数据清洗与去重：启发式规则、MinHash / SimHash、精确去重
- 5.1.3 质量过滤：分类器打分、困惑度过滤、语言识别
- 5.1.4 数据污染与隐私保护
- 5.1.5 数据混合比例与课程学习
- 5.1.6 Tokenization：BPE、WordPiece、Unigram、SentencePiece、Tiktoken
- 5.1.7 词表设计与多语言扩展

### 5.2 预训练目标
- 5.2.1 自回归语言模型（Causal LM）
- 5.2.2 掩码语言模型（MLM）、Span Corruption
- 5.2.3 Prefix LM、Fill-in-the-Middle（FIM）
- 5.2.4 多去噪混合（UL2）
- 5.2.5 多任务预训练

### 5.3 Scaling Laws
- 5.3.1 Kaplan 定律与 Chinchilla 定律
- 5.3.2 计算最优分配（参数 × 数据）
- 5.3.3 涌现能力（Emergent Abilities）与争议
- 5.3.4 数据墙与合成数据的作用

### 5.4 分布式训练
- 5.4.1 数据并行：DP、DDP、FSDP
- 5.4.2 张量并行（Megatron-style TP）
- 5.4.3 流水线并行（GPipe、1F1B、Interleaved）
- 5.4.4 序列并行、专家并行、上下文并行
- 5.4.5 ZeRO-1 / 2 / 3
- 5.4.6 3D / 4D 并行组合
- 5.4.7 混合精度训练：FP16、BF16、FP8
- 5.4.8 梯度检查点、CPU Offload
- 5.4.9 训练框架：DeepSpeed、Megatron-LM、Megatron-Core、Colossal-AI、TorchTitan

### 5.5 模型架构进阶
- 5.5.1 MoE（Mixture of Experts）：Switch Transformer、GShard、DeepSeek-MoE
- 5.5.2 稠密模型 vs 稀疏模型
- 5.5.3 路由策略：Top-k、Expert Choice、共享专家
- 5.5.4 超长上下文架构（YARN、PI、位置外推）
- 5.5.5 多 Token 预测（MTP）

---

## 第六部分 后训练与对齐

### 6.1 监督微调（SFT）
- 6.1.1 指令微调（Instruction Tuning）
- 6.1.2 多轮对话数据构造与格式
- 6.1.3 指令数据集：FLAN、Alpaca、Self-Instruct、WizardLM、Evol-Instruct
- 6.1.4 数据质量 vs 数量的权衡

### 6.2 基于人类反馈的强化学习（RLHF）
- 6.2.1 奖励模型（Reward Model）训练
- 6.2.2 PPO 算法在 LLM 上的应用
- 6.2.3 KL 约束与参考模型
- 6.2.4 InstructGPT 完整流程
- 6.2.5 奖励黑客（Reward Hacking）与缓解

### 6.3 偏好优化新方法
- 6.3.1 DPO（Direct Preference Optimization）
- 6.3.2 IPO、KTO、ORPO、SimPO、CPO
- 6.3.3 RLAIF、Constitutional AI、RLCF
- 6.3.4 GRPO（DeepSeek 系列）与强化学习推理范式
- 6.3.5 在线 vs 离线偏好学习

### 6.4 推理能力增强
- 6.4.1 Chain-of-Thought 微调
- 6.4.2 推理模型：o1 / o3、DeepSeek-R1、Claude 扩展思考
- 6.4.3 过程监督（PRM）vs 结果监督（ORM）
- 6.4.4 自我博弈、自我改进（Self-Play、STaR）
- 6.4.5 Test-Time Compute 与推理 scaling

### 6.5 参数高效微调（PEFT）
- 6.5.1 LoRA、QLoRA、DoRA、LoRA+
- 6.5.2 Adapter、Prefix Tuning、P-Tuning v1 / v2
- 6.5.3 Prompt Tuning、IA³
- 6.5.4 多 LoRA 合并与路由
- 6.5.5 PEFT 工具库：Hugging Face PEFT、Unsloth

---

## 第七部分 提示工程（Prompt Engineering）

### 7.1 基础提示设计
- 7.1.1 Zero-shot、One-shot、Few-shot
- 7.1.2 角色设定、任务描述、输出格式约束
- 7.1.3 示例选择与顺序效应

### 7.2 高级提示技术
- 7.2.1 Chain-of-Thought（CoT）与 Zero-shot CoT
- 7.2.2 Self-Consistency
- 7.2.3 Tree of Thoughts、Graph of Thoughts
- 7.2.4 ReAct（推理 + 行动）
- 7.2.5 Reflexion、Self-Refine、Self-Critique
- 7.2.6 Least-to-Most、Plan-and-Solve、Skeleton-of-Thought
- 7.2.7 Meta-Prompting、自动提示生成（APE）

### 7.3 工程实践
- 7.3.1 提示模板与变量管理
- 7.3.2 提示版本控制与 A/B 测试
- 7.3.3 提示注入攻击与防御
- 7.3.4 结构化输出（JSON Mode、Tool Use、约束解码）

---

## 第八部分 检索增强生成（RAG）

### 8.1 基础 RAG
- 8.1.1 文档加载与解析（PDF、Office、HTML、代码）
- 8.1.2 文本分块策略：固定窗口、递归、语义分块、Late Chunking
- 8.1.3 Embedding 模型选择与评估（MTEB 榜单）
- 8.1.4 向量数据库：FAISS、Milvus、Chroma、Pinecone、Qdrant、Weaviate、pgvector
- 8.1.5 检索方式：稠密、稀疏（BM25）、混合检索
- 8.1.6 Re-ranking：Cross-Encoder、ColBERT、Cohere Rerank

### 8.2 进阶 RAG
- 8.2.1 查询改写：HyDE、Multi-Query、Step-Back、Query Decomposition
- 8.2.2 Parent-Child、Small-to-Big、Auto-Merging
- 8.2.3 GraphRAG 与知识图谱融合
- 8.2.4 Self-RAG、Corrective RAG（CRAG）
- 8.2.5 Agentic RAG、迭代检索
- 8.2.6 长文档 / 长上下文时代的 RAG 价值与定位

### 8.3 RAG 评估
- 8.3.1 检索评估：Recall@k、MRR、nDCG
- 8.3.2 生成评估：忠实性、相关性、完整性
- 8.3.3 评估工具：RAGAS、TruLens、DeepEval

---

## 第九部分 智能体（Agent）

### 9.1 Agent 基础
- 9.1.1 Agent 定义与组成：感知、规划、记忆、工具
- 9.1.2 ReAct 范式与思维-行动循环
- 9.1.3 Function Calling / Tool Use 原理与实现

### 9.2 Agent 架构
- 9.2.1 单智能体：AutoGPT、BabyAGI、Voyager
- 9.2.2 多智能体：AutoGen、CrewAI、MetaGPT、LangGraph
- 9.2.3 规划-执行-反思循环（Plan-Act-Reflect）
- 9.2.4 Code Agent（OpenDevin、SWE-Agent）
- 9.2.5 Computer Use / Browser Agent（Claude Computer Use、Operator）
- 9.2.6 具身 Agent 与机器人

### 9.3 记忆与学习
- 9.3.1 短期记忆、长期记忆、情景记忆、语义记忆
- 9.3.2 向量化记忆与检索
- 9.3.3 经验重放与技能库
- 9.3.4 MemGPT、Generative Agents

### 9.4 协议与生态
- 9.4.1 MCP（Model Context Protocol）
- 9.4.2 A2A（Agent-to-Agent）协议
- 9.4.3 LangChain、LangGraph、LlamaIndex、Haystack
- 9.4.4 Agent 安全与权限控制（沙箱、最小权限）

---

## 第十部分 多模态大模型

### 10.1 视觉-语言模型
- 10.1.1 对比学习：CLIP、ALIGN、SigLIP
- 10.1.2 图文生成：BLIP、BLIP-2、InstructBLIP
- 10.1.3 开源 VLM：LLaVA 系列、MiniGPT-4、InternVL、Qwen-VL
- 10.1.4 闭源 VLM：GPT-4V / 4o、Claude Vision、Gemini
- 10.1.5 视觉编码器：ViT、Swin Transformer、视觉 Tokenizer

### 10.2 图像生成
- 10.2.1 VAE、GAN 基础
- 10.2.2 扩散模型原理：DDPM、DDIM、Score-based、Flow Matching
- 10.2.3 Latent Diffusion 与 Stable Diffusion 系列
- 10.2.4 DiT（Diffusion Transformer）
- 10.2.5 商业产品：DALL-E、Midjourney、Imagen、FLUX、Ideogram
- 10.2.6 可控生成：ControlNet、IP-Adapter、LoRA for Diffusion

### 10.3 音频与语音
- 10.3.1 语音识别：Whisper、SenseVoice
- 10.3.2 语音合成：VALL-E、XTTS、CosyVoice、Tortoise
- 10.3.3 音乐生成：MusicGen、Suno、Udio
- 10.3.4 语音对话：GPT-4o 语音模式、Moshi

### 10.4 视频生成与理解
- 10.4.1 视频生成：Sora、Runway Gen-3、Veo、可灵、Pika
- 10.4.2 时空注意力与视频扩散
- 10.4.3 视频理解：Video-LLaMA、VideoChat

### 10.5 统一多模态与具身
- 10.5.1 原生多模态模型（GPT-4o、Gemini、Claude）
- 10.5.2 Any-to-Any 架构
- 10.5.3 视觉-语言-动作（VLA）模型：RT-2、π0
- 10.5.4 世界模型（World Models）

---

## 第十一部分 推理优化与部署

### 11.1 推理加速
- 11.1.1 KV Cache 原理与优化
- 11.1.2 PagedAttention（vLLM）
- 11.1.3 Speculative Decoding / Medusa / EAGLE
- 11.1.4 Continuous Batching、动态批处理
- 11.1.5 Prefill / Decode 分离、Chunked Prefill
- 11.1.6 Prefix Caching、前缀共享

### 11.2 模型压缩
- 11.2.1 量化基础：对称/非对称、PTQ vs QAT
- 11.2.2 权重量化：GPTQ、AWQ、GGUF、bitsandbytes
- 11.2.3 激活量化：SmoothQuant、FP8
- 11.2.4 剪枝：结构化、非结构化、Wanda、SparseGPT
- 11.2.5 知识蒸馏：Soft Label、Hard Label、中间层蒸馏
- 11.2.6 低秩分解、推测解码

### 11.3 推理框架
- 11.3.1 服务端：vLLM、TensorRT-LLM、TGI、SGLang、DeepSpeed-Inference
- 11.3.2 本地端：llama.cpp、Ollama、LM Studio、MLX、CoreML
- 11.3.3 移动端：MNN、NCNN、ExecuTorch
- 11.3.4 硬件：NVIDIA GPU、AMD ROCm、Google TPU、华为昇腾、Apple Silicon

### 11.4 部署工程
- 11.4.1 模型服务化：REST、gRPC、WebSocket 流式输出
- 11.4.2 负载均衡、横向扩展、弹性伸缩
- 11.4.3 K8s、Ray Serve、KServe
- 11.4.4 成本优化：多租户、共享显存、Spot 实例
- 11.4.5 可观测性：日志、指标、链路追踪

---

## 第十二部分 评估与基准

### 12.1 通用能力
- 12.1.1 知识类：MMLU、MMLU-Pro、GPQA、CMMLU、C-Eval
- 12.1.2 推理类：BBH、ARC、HellaSwag、GSM8K、MATH、AIME
- 12.1.3 代码类：HumanEval、MBPP、LiveCodeBench
- 12.1.4 多语言：XNLI、Flores

### 12.2 长文本与检索
- 12.2.1 Needle in a Haystack（大海捞针）
- 12.2.2 RULER、LongBench、∞-Bench
- 12.2.3 多跳推理评测

### 12.3 对齐与安全
- 12.3.1 TruthfulQA、ToxiGen、RealToxicityPrompts
- 12.3.2 HHH（Helpful / Honest / Harmless）
- 12.3.3 越狱评测集

### 12.4 Agent 与工具使用
- 12.4.1 SWE-bench、SWE-bench Verified
- 12.4.2 WebArena、VisualWebArena
- 12.4.3 τ-bench、GAIA、AgentBench

### 12.5 人类评估与裁判模型
- 12.5.1 Chatbot Arena（LMSYS Elo）
- 12.5.2 MT-Bench、AlpacaEval
- 12.5.3 LLM-as-a-Judge 偏差与校准

---

## 第十三部分 安全、对齐与伦理

### 13.1 AI 安全基础
- 13.1.1 幻觉（Hallucination）机制与缓解
- 13.1.2 偏见与公平性
- 13.1.3 越狱（Jailbreak）与提示注入（Prompt Injection）
- 13.1.4 训练数据污染与模型隐私泄露
- 13.1.5 欺骗性对齐（Deceptive Alignment）

### 13.2 可解释性
- 13.2.1 机制可解释性（Mechanistic Interpretability）
- 13.2.2 Sparse Autoencoders（SAE）与特征发现
- 13.2.3 激活引导（Activation Steering）
- 13.2.4 影响函数、归因方法

### 13.3 红队与防御
- 13.3.1 红队测试（Red Teaming）方法论
- 13.3.2 对抗训练
- 13.3.3 输入 / 输出内容审核
- 13.3.4 监控、审计与应急响应

### 13.4 治理与合规
- 13.4.1 主要法规：EU AI Act、NIST AI RMF、中国生成式 AI 管理办法
- 13.4.2 版权与训练数据合规
- 13.4.3 模型卡（Model Card）与透明度报告
- 13.4.4 前沿模型安全承诺与 RSP（负责任扩展政策）

---

## 第十四部分 垂直领域应用

### 14.1 代码与软件工程
- 14.1.1 代码补全：Copilot、Cursor、Cody
- 14.1.2 Agentic 编程：Claude Code、Devin、OpenHands
- 14.1.3 代码理解、审查、调试、测试生成
- 14.1.4 端到端软件工程自动化

### 14.2 知识工作与办公
- 14.2.1 文档、邮件、会议纪要
- 14.2.2 表格与数据分析（Claude for Excel 类产品）
- 14.2.3 演示与创意（PPT、设计）
- 14.2.4 桌面 Agent（Cowork、Computer Use）

### 14.3 行业应用
- 14.3.1 教育：个性化学习、答疑、内容生成
- 14.3.2 医疗：辅助诊断、医学文献、合规挑战
- 14.3.3 法律：合同审查、案例检索、法律研究
- 14.3.4 金融：研究报告、风控、量化辅助
- 14.3.5 科研：AI for Science（AlphaFold、材料、蛋白质）
- 14.3.6 客服与营销

### 14.4 浏览器与自动化
- 14.4.1 浏览器 Agent（Claude in Chrome、Operator）
- 14.4.2 RPA 与 AI 融合
- 14.4.3 工作流自动化（n8n、Zapier + LLM）

---

## 第十五部分 实战项目路线

### 15.1 入门项目
- 15.1.1 从零实现 Mini-GPT（100～300 行 PyTorch）
- 15.1.2 用 Transformer 做机器翻译或文本摘要
- 15.1.3 基于开源模型的 Prompt 应用
- 15.1.4 简易 RAG 问答系统（LangChain / LlamaIndex）

### 15.2 进阶项目
- 15.2.1 用 LoRA / QLoRA 微调 LLaMA / Qwen / Mistral
- 15.2.2 构建企业知识库 + 多源检索 Agent
- 15.2.3 多模态应用：图文问答、语音助手
- 15.2.4 本地部署量化模型（llama.cpp / Ollama）
- 15.2.5 复现 DPO / PPO 小规模实验

### 15.3 系统级项目
- 15.3.1 分布式预训练中等规模模型（1B～7B）
- 15.3.2 自建高性能推理服务（vLLM + K8s）
- 15.3.3 多智能体协作系统（代码 / 研究 / 运营场景）
- 15.3.4 端到端 AI 产品（前端 + 后端 + 模型服务 + 监控）

---

## 第十六部分 前沿方向与开放问题

- 16.1 超长上下文（1M+ Token）与稀疏检索融合
- 16.2 Test-Time Compute 与推理范式革命
- 16.3 合成数据与自我改进（Self-Improvement）
- 16.4 模型合并（Model Merging）与模型汤（Model Soup）
- 16.5 世界模型与通用物理模拟
- 16.6 具身智能与机器人基础模型
- 16.7 神经符号结合、可验证推理
- 16.8 新架构探索：Mamba 系、RWKV、Hyena、xLSTM
- 16.9 去中心化训练与联邦学习
- 16.10 AGI / ASI 路径、对齐难题、长期安全研究

---

## 第十七部分 学习资源推荐

### 17.1 经典课程
- 17.1.1 Stanford CS224N（NLP with Deep Learning）
- 17.1.2 Stanford CS336（Language Models from Scratch）
- 17.1.3 Stanford CS229 / CS230
- 17.1.4 李宏毅机器学习与大语言模型系列
- 17.1.5 吴恩达 DeepLearning.AI 系列（含 Short Courses）
- 17.1.6 fast.ai Practical Deep Learning
- 17.1.7 Hugging Face NLP / LLM / Agents Course

### 17.2 必读论文清单
- 17.2.1 基础架构：Attention Is All You Need、BERT、GPT-1/2/3、T5
- 17.2.2 Scaling：Chinchilla、Emergent Abilities、Scaling Laws
- 17.2.3 对齐：InstructGPT、Constitutional AI、DPO
- 17.2.4 推理：CoT、ToT、ReAct、Reflexion
- 17.2.5 检索与 Agent：RAG、Toolformer、MCP
- 17.2.6 开源模型报告：LLaMA、Qwen、DeepSeek、Mistral 技术报告

### 17.3 参考书籍
- 17.3.1 《Deep Learning》— Goodfellow, Bengio, Courville
- 17.3.2 《Speech and Language Processing》— Jurafsky & Martin
- 17.3.3 《Dive into Deep Learning》— 李沐等
- 17.3.4 《大规模语言模型：从理论到实践》
- 17.3.5 《Designing Machine Learning Systems》— Chip Huyen

### 17.4 博客、社区与工具
- 17.4.1 博客：Lil'Log、Sebastian Raschka、Jay Alammar、Anthropic / OpenAI / Google Research
- 17.4.2 论文源：arXiv、Papers with Code、Hugging Face Papers
- 17.4.3 社区：Hugging Face、GitHub、知乎、Reddit r/LocalLLaMA
- 17.4.4 在线实验：Colab、Kaggle、Hugging Face Spaces
- 17.4.5 中文社区：机器之心、量子位、PaperWeekly

---

## 附录：学习建议

- **循序渐进**：基础数学与 DL → Transformer → 预训练/后训练 → 应用（RAG/Agent）→ 工程优化
- **理论结合实践**：每学一章，尽量跑通一个最小可运行示例
- **持续跟踪前沿**：每周固定时间浏览 arXiv、Hugging Face、主要实验室博客
- **建立体系**：定期复盘，用自己的语言写总结或博客
- **动手原则**：能自己实现一遍的，尽量不要只看；能跑小规模实验的，尽量不要只跑推理
