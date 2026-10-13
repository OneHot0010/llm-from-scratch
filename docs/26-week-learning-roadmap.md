# 从 Transformer 到 VLA：26 周手写复现路线

## 计划边界

- 周期：26 周，约 6 个月。
- 节奏：每周 5 个学习日 + 1 个复盘日，每天 60 分钟，共约 156 小时。
- 技术栈：Python + PyTorch；允许张量运算、自动求导、数据加载器和优化器等基础设施。
- 手写边界：不使用 `nn.MultiheadAttention`、现成 Transformer/LLaMA/VLM/VLA 主干；核心架构模块均自行实现。
- 训练目标：训练缩小模型，验证架构与学习能力，不复刻原论文的参数规模和算力规模。
- 暂不覆盖：推理引擎、量化部署、KV Cache 优化、FlashAttention、分布式训练、真实机械臂和仿真环境。
- 补漏原则：不单设 Python、PyTorch 或数学基础课；遇到阻塞时，在当日记录中增加不超过 30 分钟的定向补漏，计划整体顺延。

## 总路线

| 阶段 | 周次 | 目标 | 阶段作品 |
|---|---:|---|---|
| Transformer 与 GPT | 1-7 | 从张量级组件写到可训练的 decoder-only LM | `MiniGPT`、BPE、训练与生成闭环 |
| LLaMA | 8-12 | 掌握 RMSNorm、RoPE、SwiGLU、GQA 和 LLaMA 堆叠 | `MiniLLaMA` 与参考实现对齐 |
| 现代 Transformer 分支 | 13 | 理解长上下文与 MoE 的主流设计 | 单层稀疏 MoE 消融 |
| 视觉编码与 CLIP | 14-17 | 从 Patch Embedding 写到视觉-文本对比学习 | `TinyViT`、`TinyCLIP` |
| 多模态大模型 | 18-21 | 理解视觉 token 接入 LLM、训练阶段与评估 | `TinyLLaVA` |
| VLA | 22-26 | 掌握动作 token、动作分块与 flow action expert | `TinyOpenVLA` + 连续动作头 |

## 每日执行规则

普通学习日按 `10 分钟回忆 + 15 分钟阅读 + 25 分钟实现/实验 + 10 分钟验收` 执行。复盘日按 `15 分钟闭卷复述 + 25 分钟重写/集成 + 10 分钟验收 + 10 分钟面试` 执行。

表格中的“验收与复盘”是硬性完成条件；未通过则标记 `RETRY`，下一学习日继续，不为赶进度跳过。目录与每日文件规则见 [artifact-organization.md](artifact-organization.md)。

---

## 第一阶段：Transformer 与 GPT

### 第 1 周：模块契约与最小神经网络组件

目录：`learning/week01/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 明确批次、序列、特征维和参数维度的语义 | 配置 dataclass、随机种子、参数统计器 | 建立最小训练/测试入口，打印每层输入输出形状 | 两次同配置运行结果一致；解释 `[B,T,C]` 每一维 | [R27] 随机性与可复现文档 | [基础] 为什么只设置一个随机种子仍可能不完全可复现？ |
| D2（60m） | 线性变换与仿射变换 | `ManualLinear`，只用 `Parameter` 与矩阵乘法 | 拷贝 `nn.Linear` 权重并对齐前向、反向 | 输出误差与梯度误差均 `<1e-6`；偏置如何广播？ | [R27] autograd/gradcheck | [基础] Linear 的参数量和输入长度有关吗？ |
| D3（60m） | token id、embedding lookup 与权重矩阵关系 | `TokenEmbedding` | 与 one-hot 乘权重的结果对齐 | lookup 与 one-hot 结果一致；重复 token 梯度正确累加 | [R01] §3.4 | [基础] Embedding 和无偏置 Linear 在数学上有何关系？ |
| D4（60m） | softmax 稳定性与交叉熵 | 稳定 softmax、negative log-likelihood | 对比 `F.cross_entropy`，测试极大 logit | 无 `inf/nan`，损失与参考误差 `<1e-6` | [R01] §3.2.1；[R27] 数值稳定性 | [训练] 为什么 softmax 前要减最大值？ |
| D5（60m） | 参数初始化和梯度流的最小诊断 | 两层 MLP、梯度统计器 | 在 8 个样本上过拟合分类任务 | loss 明显下降并达到 100% 训练准确率；无零梯度异常 | [R30] §2 背景 | [训练] “模型能过拟合一个 batch”能排除哪些问题？ |
| D6 复盘（60m） | 串联本周模块契约 | 闭卷重写 Linear + Embedding | 完成形状、数值、梯度三类测试 | 不看旧代码完成；回答：何时应做参考对齐，何时应做性质测试？ | 回看本周笔记 | [基础] 如何系统测试一个新神经网络层？ |

### 第 2 周：注意力机制

目录：`learning/week02/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | Q/K/V、相似度和加权求和 | 朴素 scaled dot-product attention | 手算一个 `T=3,d=2` 样例并与代码对齐 | 手算、循环实现、矩阵实现三者一致 | [R01] §3.2.1 | [基础] Q、K、V 为什么需要不同投影？ |
| D2（60m） | `1/sqrt(d_k)` 缩放的方差动机 | 缩放版 attention | 比较不同 `d_k` 下缩放前后的分数与熵 | 能用方差说明不缩放为何易使 softmax 饱和 | [R01] §3.2.1 | [基础] 缩放因子为何不是 `1/d_k`？ |
| D3（60m） | padding mask 与 causal mask 的广播规则 | 掩码构造和 masked attention | 构造未来 token 和 padding token 干扰样例 | 被屏蔽权重为 0；全局无 `nan` | [R02] Attention 部分 | [基础] attention mask 应在 softmax 前还是后应用？ |
| D4（60m） | 多头拆分、并行子空间和输出投影 | `MultiHeadAttention` | 打印 `[B,H,T,D]` 数据流；与逐头循环对齐 | 合并结果与循环实现误差 `<1e-6` | [R01] §3.2.2 | [基础] 多头注意力为何通常不增加总投影维度？ |
| D5（60m） | 注意力能表达内容寻址 | 单层注意力复制模型 | 过拟合短序列 copy task，检查注意力矩阵 | 小批次 loss `<0.05`；注意力集中到相关位置 | [R02] 模型代码 | [训练] attention 权重可视化能否证明模型真的“理解”？ |
| D6 复盘（60m） | 闭卷推导完整 MHA | 闭卷重写 MHA 与两种 mask | 随机测试 5 组 `B,T,H,D` 组合 | 形状、因果性、梯度均通过；画出完整张量变化 | 回看 [R01] §3.2 | [基础] MHA 的时间与空间复杂度是什么？ |

### 第 3 周：Transformer Encoder

目录：`learning/week03/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | LayerNorm 的归一化维度与可学习参数 | `ManualLayerNorm` | 对齐 `nn.LayerNorm` 前向和梯度 | 随机输入误差 `<1e-5`；解释为何不跨 batch | [R01] §3.1；[R02] Norm | [基础] LayerNorm 与 BatchNorm 的核心差异是什么？ |
| D2（60m） | Position-wise FFN 的作用 | 两层 FFN + GELU/ReLU 选项 | 对比参数量和激活分布 | 输出保持 `[B,T,C]`；每个 token 独立处理 | [R01] §3.3 | [基础] FFN 为什么通常先升维再降维？ |
| D3（60m） | 残差、Pre-Norm 与 Post-Norm | 两种 residual block | 比较 8 层随机网络的梯度范数 | 能解释两种顺序对深层训练的影响 | [R01] §3.1；[R06] §2 | [训练] Pre-Norm 为什么更容易训练深层网络？ |
| D4（60m） | Encoder block 的完整数据流 | `TransformerEncoderBlock` | 单块形状、梯度和 mask 测试 | 每个参数获得有限梯度；padding 不影响有效位置 | [R01] Figure 1 | [基础] Encoder 中注意力和 FFN 分别负责什么？ |
| D5（60m） | 层堆叠与位置编码 | sinusoidal position encoding、encoder stack | 训练小型序列分类器 | 32 样本可过拟合；打乱位置编码后性能变化可解释 | [R01] §3.5 | [基础] 正弦位置编码为何能外推到更长序列？ |
| D6 复盘（60m） | 从输入到 encoder 输出闭环 | 闭卷重写一个 Encoder block | 消融残差、Norm、位置编码各一次 | 写出三项消融的预期与观测；周测试全通过 | 回看 [R01] §3 | [训练] Transformer 中最常见的梯度不稳定来源有哪些？ |

### 第 4 周：原始 Transformer Decoder 与 Seq2Seq

目录：`learning/week04/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | teacher forcing、右移标签和因果性 | target shift、causal self-attention | 修改未来 token，验证当前位置输出不变 | 所有前缀位置通过因果性测试 | [R01] §3.1、§5.4 | [训练] teacher forcing 的训练与推理差异是什么？ |
| D2（60m） | cross-attention 的查询与记忆来源 | `CrossAttention` | 对齐不同源/目标长度的形状 | 支持 `T_src != T_tgt`；解释 Q/K/V 来源 | [R01] Figure 1 | [基础] cross-attention 与 self-attention 的区别是什么？ |
| D3（60m） | Decoder block 三个子层 | `TransformerDecoderBlock` | 分别测试两种 mask | 源 padding mask 与目标 causal mask 均生效 | [R02] Decoder 部分 | [基础] Decoder 为什么需要两类注意力？ |
| D4（60m） | encoder-decoder 的端到端接口 | `Seq2SeqTransformer` | 建立 token embedding、输出头与 loss | logits 为 `[B,T_tgt,V]`，label shift 无错位 | [R01] §3 | [训练] 输出投影为何常与 token embedding 权重共享？ |
| D5（60m） | 自回归解码循环 | greedy decoder | 在 reverse/copy 数据上训练并生成 | 训练集近乎完全拟合；生成遇到 EOS 正确停止 | [R01] §5.1 | [基础] 自回归模型为何训练可并行、生成却顺序执行？ |
| D6 复盘（60m） | 掌握原始 Transformer 全貌 | 闭卷画 encoder-decoder 并重写 Decoder block | 用 3 个样例排查 label 泄漏 | 能指出所有 mask 的形状与作用位置 | 回看 [R01] Figure 1 | [训练] 如何检测 Seq2Seq 训练中的目标泄漏？ |

### 第 5 周：GPT 与自回归生成

目录：`learning/week05/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | decoder-only 与原始 Decoder 的差异 | GPT block 接口 | 移除 cross-attention，建立 next-token loss | logits/labels 严格错开一位，无未来泄漏 | [R03] §2；[R01] Decoder | [基础] GPT 为什么不需要 encoder？ |
| D2（60m） | token/position embedding 与上下文窗口 | learned position embedding | 同一 token 放在不同位置比较表示 | 超过最大长度时明确报错；形状正确 | [R03] §2.1 | [基础] learned 与 sinusoidal 位置编码各有何限制？ |
| D3（60m） | GPT 完整堆叠、最终 Norm、LM Head | `MiniGPT` | 参数量按公式和程序分别计算 | 两种参数量一致；所有层参数参与反向 | [R03] 模型章节 | [基础] LM Head 输出维度由什么决定？ |
| D4（60m） | greedy、temperature、top-k 采样 | 自回归 `generate` | 固定随机种子比较三种策略 | `temperature→0` 接近 greedy；top-k 不采到集合外 token | [R03] §3 | [训练] temperature 如何改变 token 分布？ |
| D5（60m） | 字符级语言建模闭环 | 字符词表与滑窗数据集 | 在短文本上训练 MiniGPT 并采样 | 能过拟合短文本，生成长度/EOS 行为正确 | [R03] 实验设置 | [训练] 训练 loss 下降但生成很差，应先排查什么？ |
| D6 复盘（60m） | 从文本到 loss 再到生成 | 闭卷重写 GPT block + generate | 以单 token、满窗口、空终止符做边界测试 | 全部边界测试通过；闭卷讲清 label shift | 回看本周代码 | [基础] Decoder-only 模型的一次训练前向包含哪些步骤？ |

### 第 6 周：Tokenizer 与语言模型数据管线

目录：`learning/week06/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | Unicode、UTF-8、字符与字节 token | byte tokenizer | 中英文、空白、emoji 往返编码 | 任意测试字符串 `decode(encode(x)) == x` | [R04] 背景；Python Unicode 文档 | [基础] 为什么 byte-level tokenizer 没有 OOV？ |
| D2（60m） | BPE 的频次合并目标 | BPE trainer | 对小语料打印前 20 次 merge | 每次选择最高频 pair，结果可复现 | [R04] Algorithm 1 | [基础] BPE 训练究竟学习了什么？ |
| D3（60m） | BPE 编码、解码和特殊 token | BPE encoder/decoder | 加入 BOS/EOS/PAD/UNK 并测试冲突 | 普通文本可逆；特殊 token id 唯一且稳定 | [R04]；[R05] §2 | [基础] special token 为什么不能当普通字符串随意处理？ |
| D4（60m） | 固定窗口、stride、packing 与 loss mask | LM dataset/collator | 比较丢弃尾部、padding、packing 的 token 利用率 | label 和 mask 对齐；padding 不计入 loss | [R05]；[R03] 数据部分 | [训练] sequence packing 如何提高训练效率，又带来什么风险？ |
| D5（60m） | tokenizer 与模型词表的契约 | 端到端文本数据管线 | 用自写 BPE 替换字符 tokenizer 训练小模型 | encode→batch→logits→loss→decode 全链路通过 | 回看 [R04]-[R05] | [训练] 更大词表对参数量、序列长度和训练有何影响？ |
| D6 复盘（60m） | 闭卷复现 BPE 核心过程 | 重写 merge 训练和 encode | 对未见词、中英文混合、重复空格做测试 | 能解释 BPE 与 SentencePiece 的差别 | [R05] §1-3 | [基础] tokenizer 为什么也是模型行为的一部分？ |

### 第 7 周：MiniGPT 训练闭环与阶段 Gate

目录：`learning/week07/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | AdamW、权重衰减与参数分组 | 可选：简化 AdamW；必须写参数分组器 | 用同一 batch 对比 Adam/AdamW 更新 | Norm/偏置不衰减；更新方向与参考一致 | [R30] | [训练] AdamW 与在 Adam loss 中加 L2 是否等价？ |
| D2（60m） | 初始化、残差路径与训练稳定性 | GPT 初始化函数、梯度裁剪 | 比较两种初始化的激活/梯度范数 | 无爆炸/消失；能解释裁剪治标不治本 | [R03] §2；[R06] 初始化描述 | [训练] 梯度裁剪应该放在 optimizer step 的什么位置？ |
| D3（60m） | checkpoint 的完整状态 | trainer、save/load checkpoint | 中断后恢复训练，与连续训练对齐 | 固定条件下恢复后的下一步 loss 一致 | PyTorch serialization 文档 | [训练] checkpoint 除模型权重外还应保存什么？ |
| D4（60m） | train/val 划分、交叉熵与困惑度 | evaluator | 记录 loss、perplexity、token accuracy | 验证集无数据泄漏；`ppl=exp(loss)` 使用条件说清 | [R03] §3 | [训练] perplexity 可以跨 tokenizer 直接比较吗？ |
| D5（60m） | 完整 MiniGPT 项目验收 | 清理配置、训练、生成 CLI | 训练一个 5M-30M 参数模型，保留曲线和样例 | 可保存、加载、续训、生成；小批次可过拟合 | 本阶段全部资料 | [训练] 给你一条异常 loss 曲线，你会按什么顺序排查？ |
| D6 Gate（60m） | 阶段闭卷答辩 | 从空文件写核心 attention 或 GPT block | 跑全量测试并做 10 分钟架构讲解 | 测试全过；随机抽 10 题答对至少 8 题，否则顺延 | 回看阶段笔记 | [基础] 从 token id 到 next-token loss 的完整数据流是什么？ |

---

## 第二阶段：LLaMA

### 第 8 周：RMSNorm、RoPE 与 SwiGLU

目录：`learning/week08/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | RMSNorm 与 LayerNorm 的统计量差异 | `RMSNorm` | 与公式/参考实现对齐前向和梯度 | 误差 `<1e-5`；解释为何不减均值 | [R06] §2；RMSNorm 论文 | [基础] RMSNorm 相比 LayerNorm 去掉了什么？ |
| D2（60m） | RoPE 将相对位置信息写入点积 | 二维旋转的最小实现 | 手算两个位置的旋转与内积 | 能推导点积只依赖相对位置的关键步骤 | [R08] §3 | [基础] RoPE 应用在 Q/K 还是 V？为什么？ |
| D3（60m） | 多频率 RoPE、cos/sin cache 和 shape | `RotaryEmbedding`、`apply_rope` | 测试奇偶维、不同长度和 dtype | norm 旋转前后不变；Q/K shape 不变 | [R08]；[R06] | [基础] RoPE 的旋转频率如何随维度变化？ |
| D4（60m） | GLU 门控与 SwiGLU | `SiLU`、`SwiGLU` | 对比 ReLU-FFN 与 SwiGLU 参数量和输出 | 解释 LLaMA 中间维度为何需调整 | [R06] §2 | [基础] SwiGLU 的门控分支起什么作用？ |
| D5（60m） | 将新组件接入 GPT block | LLaMA 风格 block v1 | 在同一小数据上比较 GPT block 与新 block | 两者均可过拟合；记录参数量/收敛差异 | [R06] §2 | [训练] 同参数预算下如何公平比较两种 FFN？ |
| D6 复盘（60m） | 闭卷复现三个 LLaMA 组件 | 重写 RMSNorm、RoPE、SwiGLU | 完成性质测试而非只做样例测试 | RoPE 保范数、RMSNorm 缩放性质等测试通过 | 回看 [R06]、[R08] | [基础] LLaMA 相对原始 Transformer 改了哪些核心部件？ |

### 第 9 周：MQA、GQA 与 LLaMA Attention

目录：`learning/week09/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | MHA、MQA、GQA 的头数关系 | 参数量/激活量计算器 | 对同一配置计算三者 K/V 参数量 | 手算与程序一致；画出 head 分组 | [R09] §2 | [基础] MQA、GQA 各自共享了什么？ |
| D2（60m） | query heads 到 KV heads 的映射 | `repeat_kv` | 与显式复制和索引实现对齐 | 每个 query head 映射到正确 KV 组 | [R09] §3 | [基础] `num_query_heads` 为何需整除 `num_kv_heads`？ |
| D3（60m） | GQA 前向及 RoPE 的应用位置 | `GroupedQueryAttention` | GQA 退化为 MHA/MQA 的测试 | 两个极端配置与对应实现数值一致 | [R09] §3；[R07] 模型章节 | [基础] GQA 在质量与内存之间做了什么折中？ |
| D4（60m） | LLaMA causal attention 完整路径 | LLaMA attention + mask | 因果性、padding、RoPE、GQA 联合测试 | 修改未来 token 不影响前缀输出 | [R07] LLaMA 2 架构 | [基础] RoPE 与 causal mask 谁先谁后？ |
| D5（60m） | 数值与梯度参考对齐 | 参考适配脚本 | 随机权重对齐一个可信 LLaMA attention | 前向、输入梯度、参数梯度均在容差内 | [R28] attention 源码，仅用于校验 | [训练] 参考对齐失败时如何定位是布局还是公式问题？ |
| D6 复盘（60m） | 讲清 GQA 数据布局 | 闭卷重写 GQA | 随机生成 5 种合法配置做性质测试 | 无硬编码头数；所有配置通过 | 回看 [R09] | [基础] GQA 是否减少训练 FLOPs？主要收益在哪里？ |

### 第 10 周：完整 MiniLLaMA

目录：`learning/week10/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 配置驱动模型与不变量校验 | `LlamaConfig` | 对非法 head、维度、长度配置做失败测试 | 错误信息明确；参数量可预估 | [R07] 配置表 | [基础] 一个 LLaMA 配置至少需要哪些字段？ |
| D2（60m） | Pre-Norm LLaMA block | `LlamaDecoderBlock` | 单层前向、反向、残差检查 | 输入输出 shape 相同；所有分支有梯度 | [R06] §2 | [基础] LLaMA block 中有几次归一化？ |
| D3（60m） | block stack、final norm 与 LM head | `MiniLLaMA` | 检查层数、权重共享和参数注册 | state dict 无漏参；logits shape 正确 | [R06]-[R07] | [基础] final norm 为什么放在所有 block 之后？ |
| D4（60m） | SentencePiece 与 LLaMA tokenizer 契约 | tokenizer adapter | 训练一个小 SentencePiece 模型并往返测试 | BOS/EOS/PAD 行为与配置一致 | [R05]；[R06] 数据部分 | [基础] SentencePiece 为什么可直接处理原始字符串？ |
| D5（60m） | 文本、mask、label 的完整 batch | LLaMA collator | 混合长短样本并运行一次训练 step | padding loss 为 0；无未来泄漏 | [R07] 训练描述 | [训练] 左 padding 和右 padding 对 causal LM 有何影响？ |
| D6 复盘（60m） | 闭卷画 MiniLLaMA | 重写 block 伪代码与 shape 表 | 用单 token、短序列、满长度测试 | 所有边界通过；可在 5 分钟内讲清架构 | 回看本周代码 | [基础] LLaMA 和 GPT-2 的架构差异清单是什么？ |

### 第 11 周：MiniLLaMA 训练

目录：`learning/week11/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 参数预算、token 预算与目标规模 | 模型规模估算器 | 设计 20M-80M 参数 MiniLLaMA 配置 | 48GB 可轻松运行；写清选择依据 | [R29] 核心结论 | [训练] 参数量、训练 token 与算力预算如何相互制约？ |
| D2（60m） | 学习率 warmup 与 cosine decay | scheduler | 绘制并检查完整学习率序列 | 首尾、warmup 边界无 off-by-one | [R06] 训练设置 | [训练] 为什么大模型训练常使用 warmup？ |
| D3（60m） | 稳定训练所需监控项 | 训练日志与梯度/激活统计 | 在小语料启动训练，记录 loss、lr、grad norm | 指标按 step 对齐，无 `nan` | [R07] 训练设置 | [训练] loss spike 可能来自哪些数据或优化问题？ |
| D4（60m） | 训练与验证差距 | evaluator + sample callback | 比较 train/val loss 并定期生成 | 样例与 checkpoint step 对应；无验证泄漏 | [R03] 评估方法 | [训练] 预训练模型如何判断开始过拟合？ |
| D5（60m） | 可恢复、可重复的训练闭环 | 完善 trainer | 完成一个短训练并从 checkpoint 续训 | 续训指标连续；保存最终配置和样例 | 本周资料 | [训练] 如何判断训练未生效是数据、模型还是优化器问题？ |
| D6 复盘（60m） | 分析一次真实训练日志 | 无新增模块 | 从曲线反推 3 个问题并用代码验证 | 给出证据链，而非只写猜测 | 回看实验日志 | [训练] 面试中如何讲清一次训练故障排查？ |

### 第 12 周：参考对齐、消融与 LLaMA Gate

目录：`learning/week12/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 配置与 state dict 映射 | 权重映射器 | 将 tiny 随机参考 LLaMA 权重载入自写模型 | 无 missing/unexpected key；逐层输出可比较 | [R28] 模型源码 | [基础] 两个实现权重名不同，如何证明其数学等价？ |
| D2（60m） | 端到端数值对齐方法 | hook/逐层对齐工具 | 对齐 embedding、首层、末层、logits | 首个误差出现位置可被准确定位 | [R28] | [训练] 为什么只对齐最终 logits 不足以快速定位错误？ |
| D3（60m） | 架构消融设计 | 配置开关 | 比较 RMSNorm/LayerNorm、SwiGLU/ReLU、GQA/MHA | 每次只改变一个变量并记录结论 | [R06]、[R09] | [训练] 如何设计可信的架构消融实验？ |
| D4（60m） | 参数、FLOPs 与序列长度关系 | 复杂度报告器 | 推导 attention/FFN 的主要计算量 | 手算和工具结果一致；说明长序列瓶颈 | [R01] Table 1 | [基础] 何时 FFN 计算量会高于 attention？ |
| D5（60m） | 架构文档能力 | 无新增代码 | 写 MiniLLaMA 架构说明、限制与测试策略 | 新读者可仅按文档复现接口 | 回看阶段资料 | [基础] 用两分钟向面试官介绍 LLaMA 架构 |
| D6 Gate（60m） | LLaMA 阶段答辩 | 闭卷重写一个 LLaMA block | 全测试、参考对齐、训练样例、10 题抽测 | 四项均通过且答题至少 80%，否则顺延 | 本阶段资料 | [训练] LLaMA 训练不稳定时你的排查顺序是什么？ |

---

## 第三阶段：现代 Transformer 分支

### 第 13 周：长上下文概念与稀疏 MoE

目录：`learning/week13/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | RoPE 长度外推问题与缩放思路 | RoPE scaling 配置接口 | 比较训练长度内外的位置相似度 | 能解释直接外推为何退化；不追求完整训练 | [R08]；LLaMA 长上下文资料 | [基础] RoPE 外推失败的直观原因是什么？ |
| D2（60m） | 长序列下 attention 的二次复杂度 | attention 内存估算器 | 对 1K/4K/16K 长度计算矩阵大小 | 结果与实际张量字节数一致 | [R01] Table 1 | [训练] 长上下文首先撞到的是参数内存还是激活内存？ |
| D3（60m） | MoE router、专家和 token routing | top-k router | 构造 token 到专家的路由结果 | 每个 token 恰有 k 个专家；概率归一 | [R10] Switch Transformer §2 | [基础] MoE 如何增加参数量却不同比增加计算量？ |
| D4（60m） | expert capacity、负载均衡和辅助损失 | sparse expert layer + aux loss | 人为制造路由塌缩并观察辅助损失 | 能检测“所有 token 去同一专家” | [R10] §2.2-2.3 | [训练] MoE 的负载不均会造成什么问题？ |
| D5（60m） | 在 LLaMA block 中替换 FFN | `LlamaMoEBlock` | 与 dense FFN 做参数、激活和小批次拟合对比 | 两者均可训练；记录专家使用率 | [R10]；Mixtral 论文架构部分 | [基础] MoE 的 expert parallel 与模型结构是什么关系？ |
| D6 复盘（60m） | 区分架构能力与系统优化 | 闭卷重写 top-k routing | 汇总长上下文/MoE 的收益、代价和未覆盖项 | 明确本计划暂不实现的加速内容 | 回看本周资料 | [基础] 为什么“参数更多”不一定意味着每 token FLOPs 更多？ |

---

## 第四阶段：视觉编码与 CLIP

### 第 14 周：图像 Patch 与 Vision Transformer

目录：`learning/week14/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 图像 `[B,C,H,W]`、归一化和 patch 网格 | image preprocess、shape checker | 对不同尺寸验证网格可整除条件 | 均值/方差和 shape 符合配置 | [R11] §3.1 | [多模态] 图像归一化为何必须匹配预训练配置？ |
| D2（60m） | patchify/unpatchify 的索引关系 | 纯张量 `patchify`/`unpatchify` | 随机图像往返重建 | 像素完全一致；支持多通道 | [R11] Figure 1 | [多模态] patch size 如何影响 token 数和计算量？ |
| D3（60m） | Patch Embedding 等价于步长卷积 | 线性 patch projector | 与同权重 `Conv2d` 数值对齐 | 输出误差 `<1e-6`，token 顺序一致 | [R11] §3.1 | [多模态] Patch Embedding 为什么可用卷积实现？ |
| D4（60m） | class token 与二维位置信息 | CLS token、learned position embedding | 打印图像 token 序列和位置索引 | token 数为 `N_patches+1`；广播正确 | [R11] §3.1 | [多模态] ViT 为什么引入 CLS token？ |
| D5（60m） | ViT encoder 与文本 encoder 的共性 | `TinyViT` | 接入自写 Transformer encoder 做分类前向 | logits 正确；每层梯度有限非零 | [R11] §3 | [多模态] ViT 与文本 Transformer 的主要差异有哪些？ |
| D6 复盘（60m） | 闭卷画出图像到 logits 的路径 | 重写 patchify + ViT 输入层 | 测试 16/32 patch size 和非法分辨率 | 能逐项解释 token 数与参数量变化 | 回看 [R11] | [多模态] 分辨率翻倍时 self-attention 计算量如何变化？ |

### 第 15 周：TinyViT 训练与分析

目录：`learning/week15/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 图像增广与 train/eval 预处理差异 | dataset transform 配置 | 检查增广前后样本和标签不变性 | 验证集无随机增广；归一化一致 | [R11] 训练细节 | [训练] 哪些图像增广可能改变标签语义？ |
| D2（60m） | 分类 head、label smoothing | classifier + 可选平滑 loss | 先过拟合 32 张图像 | 训练准确率接近 100% | [R11] Appendix | [训练] label smoothing 对置信度和泛化有何影响？ |
| D3（60m） | ViT 训练日志与学习率 | trainer 适配视觉 batch | 训练 TinyViT 短实验 | loss 下降；checkpoint 可恢复 | [R11] §4 | [训练] ViT 为何通常比 CNN 更依赖数据和正则化？ |
| D4（60m） | attention map 的能力与局限 | attention 输出 hook | 查看 CLS 到 patch 的权重分布 | shape/归一化正确；不把热图当因果解释 | [R11] Figure 6 | [多模态] attention map 能说明什么，不能说明什么？ |
| D5（60m） | patch size 消融 | 配置化 patch size | 固定其余条件比较 token 数、速度和精度 | 结论包含计算-细节权衡 | [R11] | [多模态] 更小 patch 一定更好吗？ |
| D6 复盘（60m） | ViT 阶段小验收 | 闭卷重写 PatchEmbedding | 全测试 + 小批次过拟合 + 一项消融 | 三项通过；能两分钟讲清 ViT | 回看两周资料 | [多模态] 如何把预训练 ViT 接到其他模态模型？ |

### 第 16 周：CLIP 架构与对比学习

目录：`learning/week16/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 双塔、共享空间和 batch 内负样本 | 相似度矩阵函数 | 手算 3 对图文的 cosine similarity | 结果为 `[B,B]`，正样本在对角线 | [R12] §2.1 | [多模态] CLIP 为什么采用双塔而非早期融合？ |
| D2（60m） | 图像 encoder 的统一输出接口 | image encoder adapter | 将 TinyViT 输出映射为单向量 | 任意 batch 输出 `[B,D]` 且可归一 | [R12] Figure 3 | [多模态] CLIP 训练后为何可单独使用图像塔？ |
| D3（60m） | 文本 encoder、EOT pooling | CLIP text encoder | 比较 EOT/mean/首 token pooling | padding 不影响 pooled 表示 | [R12] §2.1 | [多模态] CLIP 文本表示为何常取 EOT 位置？ |
| D4（60m） | 投影、L2 归一化与温度 | projection heads、learnable logit scale | 比较归一化前后 logit 范围 | embedding norm 为 1；温度梯度存在 | [R12] Figure 3 | [训练] CLIP 为什么要学习 temperature/logit scale？ |
| D5（60m） | 对称 InfoNCE loss | `clip_loss` | 与 image→text、text→image 两个 CE 对齐 | 两方向 loss 平均正确；匹配样本低 loss | [R12] 伪代码 | [多模态] CLIP loss 为什么是对称的？ |
| D6 复盘（60m） | 闭卷复现 CLIP 前向与 loss | 重写 similarity + loss | 构造完美匹配、错配、重复描述三类样例 | loss 排序符合预期；解释假负样本 | 回看 [R12] | [训练] batch size 为什么会影响对比学习？ |

### 第 17 周：TinyCLIP 训练与阶段 Gate

目录：`learning/week17/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 图文配对数据与数据泄漏 | paired dataset/collator | 为小图像集生成类别描述模板 | 同一原图不跨 train/val；配对索引正确 | [R12] 数据部分 | [训练] 图文数据去重为什么重要？ |
| D2（60m） | batch negatives 与重复语义 | 多正样本 mask（实验版） | 比较单正样本和多正样本 loss | 能指出重复 caption 被当负样本的问题 | [R13] SigLIP 动机 | [多模态] in-batch negative 何时会产生 false negative？ |
| D3（60m） | TinyCLIP 训练闭环 | `TinyCLIP` trainer | 训练短实验并记录 retrieval 指标 | 正配对相似度整体高于随机错配 | [R12] §3 | [训练] 对比 loss 下降但 retrieval 不升，可能是什么原因？ |
| D4（60m） | zero-shot 分类的 prompt 机制 | zero-shot classifier | 多模板文本 embedding 做类别预测 | 明确无分类 head；结果优于随机基线 | [R12] §3.1 | [多模态] CLIP 如何实现 zero-shot 分类？ |
| D5（60m） | Recall@K 与双向检索 | retrieval evaluator | 计算 I2T/T2I Recall@1/5 | 用小样例手算并与代码一致 | [R12] 评估部分 | [多模态] 图搜文和文搜图指标为何可能不同？ |
| D6 Gate（60m） | 视觉/CLIP 阶段答辩 | 闭卷重写 CLIP loss | 跑 ViT/CLIP 测试、过拟合与检索评估 | 四项通过；随机 10 题答对至少 8 题 | 本阶段资料 | [多模态] ViT、CLIP 和 VLM 三者的接口如何衔接？ |

---

## 第五阶段：多模态大模型

### 第 18 周：视觉 Token 接入语言模型

目录：`learning/week18/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 早期融合、cross-attention、projector 三类路线 | 架构接口草图 | 对 LLaVA/Flamingo 做数据流对比 | 写出各路线训练成本和适用场景 | [R15] §3；[R16] §2 | [多模态] 视觉信息接入 LLM 有哪些主流方式？ |
| D2（60m） | vision hidden size 到 LM hidden size 映射 | linear/MLP projector | 对随机视觉特征测试映射和梯度 | 输出 `[B,N_img,D_lm]`；梯度回传策略明确 | [R15] §3.1 | [多模态] projector 看似简单，为何仍能有效对齐模态？ |
| D3（60m） | image placeholder 与视觉 token 插入 | multimodal sequence builder | 将 `<image>` 替换为 N 个视觉 embedding | token 顺序、position、attention mask 均正确 | [R15] §3.1 | [多模态] 一个 image token 为何会展开成多个 patch token？ |
| D4（60m） | 多模态 label mask | multimodal collator | 只对 assistant 文本计算 loss | 图像、用户 prompt、padding 标签均为 ignore id | [R15] 训练细节 | [训练] 多模态 SFT 中哪些 token 通常不计算 loss？ |
| D5（60m） | 视觉 encoder + projector + LLM 联合前向 | `TinyVLM` 骨架 | 随机图像/文本完成前向反向 | 三段接口 shape 正确；冻结策略可配置 | [R18] §3 | [多模态] 冻结视觉塔和 LLM 时谁还能学习？ |
| D6 复盘（60m） | 闭卷画多模态 token 序列 | 重写 sequence builder | 测试图片在句首/句中、不同文本长度 | 无错位；能解释 position id 处理 | 回看本周资料 | [多模态] 多模态序列最容易出现哪些 mask/对齐 bug？ |

### 第 19 周：LLaVA 类架构

目录：`learning/week19/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | LLaVA 的视觉塔、projector、LLM 三段式结构 | `TinyLLaVAConfig` | 映射到自写 TinyViT + MiniLLaMA | 每个配置字段能对应论文组件 | [R15] §3 | [多模态] LLaVA 的核心结构为何如此简单？ |
| D2（60m） | 预训练视觉特征与从零训练的差异 | vision tower adapter | 比较冻结/不冻结时梯度与可训练参数量 | 参数统计准确；冻结模块无梯度 | [R15] §3.1；[R14] 摘要 | [训练] 为什么第一阶段常冻结视觉塔和 LLM？ |
| D3（60m） | 特征对齐阶段的训练目标 | projector-only training step | 在合成图文对上只更新 projector | 仅 projector 权重变化；loss 可下降 | [R15] §3.2 | [多模态] projector 预训练在对齐什么？ |
| D4（60m） | instruction tuning 对输入输出格式的要求 | conversation formatter | 构造 system/user/image/assistant 序列 | assistant 之外标签全部屏蔽 | [R15] §3.2 | [多模态] 多轮对话中 loss mask 应如何设置？ |
| D5（60m） | 完整 TinyLLaVA 前向与生成 | `TinyLLaVA` | 图像条件下生成短答案 | 改变图像时视觉 token 及输出分布发生变化 | [R15] | [多模态] 如何验证模型没有完全忽略图像？ |
| D6 复盘（60m） | 闭卷重写 TinyLLaVA 数据流 | 重写 projector + 拼接逻辑 | 逐层打印 shape，排查一个故意注入的错位 bug | 能定位并解释 bug 根因 | 回看本周代码 | [多模态] LLaVA 与 CLIP 的训练目标和输出有何区别？ |

### 第 20 周：多模态数据与训练

目录：`learning/week20/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 可控合成数据用于能力验证 | 几何图形图像+问答生成器 | 生成颜色、数量、位置三类问题 | 图像、答案和元数据一致；划分无泄漏 | [R15] 数据格式 | [训练] 为什么先用合成任务验证多模态管线？ |
| D2（60m） | 变长文本与固定/变长视觉 token batching | multimodal collator | 混合不同问题长度并构造 batch | attention/label mask 与每个样例一致 | [R18] 数据管线 | [多模态] batch 中视觉 token 数不同时如何处理？ |
| D3（60m） | 两阶段训练与灾难性遗忘 | trainer phase config | 明确 projector-only 与 instruction-tuning 参数组 | 每阶段可训练参数符合预期 | [R15] §3.2 | [训练] 全量微调为什么可能破坏原模型能力？ |
| D4（60m） | 对齐阶段小实验 | 无新增模块 | 训练 projector，使图像条件影响答案 | 训练集可过拟合；随机打乱图像后性能下降 | [R15] 实验部分 | [多模态] 如何证明视觉-语言对齐真的发生？ |
| D5（60m） | 指令微调和生成评估 | generation evaluator | 微调 TinyLLaVA，保留三类问题输出 | 独立验证集优于文本先验基线 | [R15] §4 | [训练] 多模态模型 loss 正常但回答模板化，如何排查？ |
| D6 复盘（60m） | 复盘训练证据链 | 无新增模块 | 汇总 loss、准确率、打乱图像消融、失败样例 | 至少一项结论由对照实验支持 | 回看实验日志 | [多模态] 多模态训练应保留哪些定性和定量证据？ |

### 第 21 周：现代 VLM 分支、评估与 Gate

目录：`learning/week21/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | Flamingo gated cross-attention 与 media token | 最小 cross-attention resampler 实验 | 与 LLaVA token 拼接做 shape/计算对比 | 写出两种路线优缺点，不要求完整复现 | [R16] §2 | [多模态] Flamingo 为什么使用 cross-attention 而非直接拼接？ |
| D2（60m） | 动态分辨率与视觉 token 压缩 | 2x2 patch merger | 不同长宽图像计算 token 数并合并 | 保留二维顺序；合并后 token 数符合公式 | [R17] §2 | [多模态] 动态分辨率解决了固定缩放的什么问题？ |
| D3（60m） | SigLIP、DINOv2 与视觉特征互补 | 双视觉特征融合器 | 拼接两路 TinyViT 特征并投影 | 维度、归一化和 token 对齐明确 | [R13]-[R14]；[R18] | [多模态] 对比学习特征与自监督视觉特征可能如何互补？ |
| D4（60m） | 幻觉、语言先验与图像依赖评估 | 图像打乱/遮挡评估器 | 比较正确图、错图、空图三组答案 | 指标能揭示模型是否忽略图像 | [R15] 评估；VLM hallucination 综述选读 | [多模态] 如何定量判断 VLM 的回答依赖图像？ |
| D5（60m） | TinyLLaVA 工程验收 | 清理模型、训练、生成与测试入口 | 从新环境按文档跑通最小样例 | 可前向、反向、保存、加载、生成 | 本阶段全部资料 | [训练] VLM 的端到端测试应覆盖哪些接口？ |
| D6 Gate（60m） | 多模态阶段答辩 | 闭卷画 LLaVA/Flamingo 数据流 | 跑测试、过拟合、打乱图像消融与 10 题抽测 | 四项通过且答题至少 80%，否则顺延 | 本阶段资料 | [多模态] 从图像像素到答案 token 的完整路径是什么？ |

---

## 第六阶段：Vision-Language-Action

### 第 22 周：机器人序列数据与动作表示

目录：`learning/week22/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 模仿学习中的 observation、state、instruction、action | VLA sample dataclass | 用 shape 表描述一个 episode 和一个训练样本 | 区分 episode/time step/window；接口无歧义 | [R19] §2；[R20] §3 | [VLA] VLA 与普通 VLM 的输入输出差异是什么？ |
| D2（60m） | 末端位姿增量、关节动作、夹爪和坐标系 | action schema + normalizer | 构造 7D 动作，做 normalize/denormalize | 往返误差可忽略；每一维物理含义明确 | [R22] §3.2 | [VLA] 为什么动作归一化必须按数据集/机器人统计？ |
| D3（60m） | episode window、时间对齐与数据泄漏 | trajectory window dataset | 从合成 episode 提取 `(obs_t, action_t:t+h)` | 不跨 episode；时间索引无越界/错位 | [R19] 数据格式；[R33] dataset docs | [训练] 机器人序列数据最常见的时间对齐 bug 是什么？ |
| D4（60m） | 分位数离散化与动作 token | action tokenizer | 逐维量化 7D 动作并反量化 | token 范围正确；报告量化 MAE 和边界裁剪率 | [R21] 动作 token；[R22] §3.2 | [VLA] 连续动作离散成 token 的收益与损失是什么？ |
| D5（60m） | 无仿真条件下的可控离线验证 | 合成视觉-指令-动作数据生成器 | 生成颜色/位置决定动作的 toy dataset | 存在可学习规律、train/val 独立、基线明确 | [R22] 训练形式 | [训练] toy dataset 怎样设计才不只是记忆样本 ID？ |
| D6 复盘（60m） | 闭卷描述 VLA 样本到 loss | 重写 action tokenizer 核心逻辑 | 检查极值、常量维、越界动作 | 所有边界通过；能解释离线指标局限 | 回看本周资料 | [VLA] action token accuracy 高是否必然控制效果好？ |

### 第 23 周：TinyOpenVLA 架构

目录：`learning/week23/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | OpenVLA 的双视觉塔、projector、LLaMA、动作 token | 架构配置与数据流图 | 将真实组件映射为 Tiny 组件 | 每段输入输出 shape 和冻结策略明确 | [R22] §3；[R18] | [VLA] OpenVLA 相比 LLaVA 改变了什么？ |
| D2（60m） | SigLIP/DINOv2 特征融合思想 | 双 TinyViT fusion encoder | 拼接两路 patch 特征并处理 token 对齐 | 输出 token 数/维度正确；两路都有梯度或明确冻结 | [R13]-[R14]；[R22] §3.1 | [VLA] OpenVLA 为什么融合两类视觉 encoder？ |
| D3（60m） | 动作 token 占用词表与解码契约 | VLA vocabulary adapter | 建立动作 bin 到保留 token id 的双向映射 | 无文本 token 冲突；7 维顺序稳定 | [R22] §3.2 | [VLA] 动作 token 应如何加入或复用 LLM 词表？ |
| D4（60m） | 图像、指令、动作的训练序列与 label mask | OpenVLA sequence builder | 构造完整输入并只监督动作位置 | loss 位置数量与动作维度一致 | [R22] §3.2 | [训练] 若把 prompt token 也计入 loss，会发生什么？ |
| D5（60m） | TinyOpenVLA 端到端前向 | `TinyOpenVLA` | 图像+指令→动作 logits→解码动作 | 可前向反向；改变图像/指令会改变 logits | [R22] Figure 2 | [VLA] 从一个观测到 7D 动作经历哪些模块？ |
| D6 复盘（60m） | 闭卷重写动作序列构造 | 重写 VLA forward 伪代码 | 故意交换动作维度，依靠测试捕获 | 测试能失败并定位；架构图可闭卷完成 | 回看 [R22] | [VLA] VLA 中哪个接口最容易造成“能训练但动作错位”？ |

### 第 24 周：TinyOpenVLA 训练与离线评估

目录：`learning/week24/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | Open X/LeRobot 数据 schema 与 embodiment 差异 | dataset adapter interface | 把 toy 数据转换为统一 episode schema | 字段、时间戳、相机、state/action shape 可验证 | [R19]；[R33] | [VLA] 跨机器人数据混训为何需要统一 schema？ |
| D2（60m） | 全量微调、冻结和 LoRA 的取舍 | 自写 `LoRALinear` | 对 projector/LLM 三种策略统计可训练参数 | LoRA 合并前后输出对齐；参数统计正确 | [R31]；[R22] §5.3 | [训练] LoRA 的低秩矩阵插在什么位置，为什么有效？ |
| D3（60m） | 动作 token 的监督训练 | VLA trainer | 在 toy dataset 上训练 TinyOpenVLA | 能过拟合小批次；token accuracy 显著高于随机 | [R22] §3.2 | [VLA] VLA 的交叉熵 loss 与语言模型 loss 有何异同？ |
| D4（60m） | token 解码、反归一化与动作约束 | action postprocessor | 输出连续动作并检查合法范围 | 无越界/维度错序；往返流程可追踪 | [R22] 推理流程 | [VLA] 模型输出到机器人控制器之间还需哪些处理？ |
| D5（60m） | 离线 token accuracy、MAE、每维指标 | offline evaluator | 在独立数据上比较随机/均值/模型基线 | 指标手算对齐；报告每维误差与整体误差 | [R22] 评估方法 | [VLA] 为什么只报告平均 MAE 可能掩盖严重问题？ |
| D6 Gate（60m） | OpenVLA 路线小验收 | 无新增模块 | 测试、过拟合、离线评估、架构答辩 | 四项通过；明确离线好不等于闭环控制好 | 回看三周资料 | [VLA] behavioral cloning 的 distribution shift 问题是什么？ |

### 第 25 周：动作分块与 Flow Matching Action Expert

目录：`learning/week25/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 单步动作、action chunk 与闭环频率 | action chunk dataset/head | 比较 horizon 1/4/8 的 label shape 与预测目标 | 不跨 episode；明确执行长度与预测长度区别 | [R24] §3；[R26] §3 | [VLA] action chunking 的优势和代价是什么？ |
| D2（60m） | 连续回归基线与多峰动作问题 | MLP chunk regression head | 在双峰 toy action 上训练 MSE baseline | 观察均值化现象并保存证据 | [R23] §1-3 | [VLA] MSE 为什么会在多模态动作上预测“平均动作”？ |
| D3（60m） | diffusion/flow matching 的路径、噪声和速度场 | 线性概率路径采样器 | 手算并验证 `x_t=(1-t)x_0+t x_1` 及目标速度 | 采样 shape/边界正确；能解释训练目标 | [R25] 方法；[R26] §3.1 | [VLA] flow matching 与直接回归的学习目标有何不同？ |
| D4（60m） | 条件 action expert 与时间 embedding | velocity network/action expert | 以 VLM feature、state、noisy action、t 为输入 | 四类条件都影响输出；梯度路径清楚 | [R26] §3.1 | [VLA] action expert 为什么还需要机器人 state？ |
| D5（60m） | flow loss 与 ODE 迭代采样 | flow trainer + Euler sampler | 在 toy action chunk 上训练并采样 | 生成分布优于 MSE 均值基线；无仿真 | [R26] §3；[R25] | [VLA] flow policy 推理为何需要多步迭代？ |
| D6 复盘（60m） | 比较 token action 与连续 action expert | 闭卷画 OpenVLA/SmolVLA 风格路径 | 从精度、延迟、表达能力、训练复杂度做表格 | 给出有条件的技术选择，不下绝对结论 | [R22]、[R26] | [VLA] 何时选动作 token，何时选 flow action head？ |

### 第 26 周：综合项目与最终 Gate

目录：`learning/week26/`

| 日程 | 理论目标 | 手写模块 | 实验任务 | 验收与复盘 | 阅读 | 面试问题 |
|---|---|---|---|---|---|---|
| D1（60m） | 固化最终项目范围与接口 | capstone config/API | 选择同一视觉-语言骨干上的 token head + flow head | 写出模型图、shape 表、数据 schema、指标和风险 | 回看全路线架构图 | [综合] 如何把一个研究想法拆成可验证的最小实现？ |
| D2（60m） | 端到端训练的可复现性 | 统一 train/eval CLI | 从干净配置训练两个动作头 | seed、配置、checkpoint、日志齐全 | 回看训练规范 | [训练] 如何保证两个动作头的比较公平？ |
| D3（60m） | 测试与消融覆盖 | 完善全链路测试 | 做图像打乱、指令打乱、state 移除三项消融 | 每项消融有假设、结果和解释 | 回看 Gate 标准 | [VLA] 如何证明模型同时使用了视觉、语言和状态？ |
| D4（60m） | 全栈架构串联 | 无新增模块 | 闭卷画 Transformer→LLaMA→VLM→VLA 演进图 | 15 分钟内讲清每次架构变化的动机 | 全路线笔记 | [综合] 从 Transformer 到 VLA，哪些组件复用，哪些是新增？ |
| D5（60m） | 技术报告与项目可读性 | 清理 README、模型卡、限制说明 | 汇总指标、失败样例、消融和后续工作 | 新读者能按 README 复现实验；不夸大离线结果 | [R22]、[R26] limitations | [综合] 面试中如何诚实而有说服力地介绍项目局限？ |
| D6 Final Gate（60m） | 最终答辩与下一阶段决策 | 随机抽一个核心模块闭卷重写 | 跑完整测试；抽 20 道跨阶段题；演示端到端预测 | 测试全过、答题至少 80%、演示可复现；否则定向补周 | 全部资料 | [综合] 设计一个从数据到部署的 VLA 系统，你会如何分层？ |

---

## 最终应交付的作品

1. `MiniGPT`：自写 attention、block、BPE、训练、checkpoint 和生成。
2. `MiniLLaMA`：自写 RMSNorm、RoPE、SwiGLU、GQA，并与参考实现逐层对齐。
3. `LlamaMoEBlock`：具备 top-k routing 与负载均衡统计。
4. `TinyViT`：自写 patch、视觉 Transformer、训练与消融。
5. `TinyCLIP`：双塔、对称对比损失、zero-shot 和 retrieval 评估。
6. `TinyLLaVA`：视觉 projector、多模态序列、两阶段训练和图像依赖消融。
7. `TinyOpenVLA`：动作 tokenizer、统一数据 schema、动作监督和离线评估。
8. `FlowActionExpert`：动作分块、条件 flow matching、迭代采样和回归基线比较。
9. 一份综合技术报告和一套覆盖基础、训练、多模态、VLA 的面试题库。

## 26 周后的建议

本计划完成后，再按目标选择后续分支，不提前挤占当前日程：

- **训练方向**：混合精度、梯度检查点、FSDP/ZeRO、数据治理和大规模实验设计。
- **推理方向**：KV Cache、FlashAttention、量化、continuous batching 和推理引擎。
- **VLM 方向**：动态分辨率、多图/视频、OCR、grounding 和更系统的幻觉评估。
- **VLA 方向**：LeRobot/真实数据微调、LIBERO 等仿真评估、异步执行、真实机械臂闭环。
- **研究方向**：OpenVLA-OFT、SmolVLA、π0 系列以及更新的 action expert 路线。

## 阅读材料索引

| 编号 | 材料 |
|---|---|
| R01 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| R02 | [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) |
| R03 | [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| R04 | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) |
| R05 | [SentencePiece](https://arxiv.org/abs/1808.06226) |
| R06 | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) |
| R07 | [Llama 2](https://arxiv.org/abs/2307.09288) 与 [Llama 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) |
| R08 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |
| R09 | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) |
| R10 | [Switch Transformers](https://arxiv.org/abs/2101.03961) 与 [Mixtral of Experts](https://arxiv.org/abs/2401.04088) |
| R11 | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) |
| R12 | [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) |
| R13 | [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) |
| R14 | [DINOv2](https://arxiv.org/abs/2304.07193) |
| R15 | [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) |
| R16 | [Flamingo](https://arxiv.org/abs/2204.14198) |
| R17 | [Qwen2-VL](https://arxiv.org/abs/2409.12191) |
| R18 | [Prismatic VLMs](https://arxiv.org/abs/2402.07865) |
| R19 | [Open X-Embodiment](https://arxiv.org/abs/2310.08864) |
| R20 | [RT-1](https://arxiv.org/abs/2212.06817) |
| R21 | [RT-2](https://arxiv.org/abs/2307.15818) |
| R22 | [OpenVLA](https://arxiv.org/abs/2406.09246) 与 [官方项目](https://openvla.github.io/) |
| R23 | [Diffusion Policy](https://arxiv.org/abs/2303.04137) |
| R24 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705) |
| R25 | [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) |
| R26 | [SmolVLA](https://arxiv.org/abs/2506.01844) |
| R27 | [PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html) |
| R28 | [Transformers LLaMA implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)，仅用于完成自写实现后的对齐 |
| R29 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) |
| R30 | [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) |
| R31 | [LoRA](https://arxiv.org/abs/2106.09685) |
| R32 | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) |
| R33 | [LeRobot Datasets](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) |
