# LLM & Agent & 生成式推荐 后训练面试准备指南

> 基于 efficient-BEAR (Beam-Search-Aware Regularization) 项目深度解析
> 适用于：LLM算法实习面试准备

---

## 第一部分：项目深度理解

### 1.1 项目背景与核心问题

**BEAR是什么？**
BEAR (Beam-Search-Aware Regularization) 是发表于 SIGIR 2026 的论文，解决LLM用于推荐系统时训练与推理不一致的问题。

**核心问题：**
- 训练时：使用 teacher forcing，每一步都基于ground truth token
- 推理时：使用 beam search，每一步基于模型自己的预测
- 这种不一致性导致训练目标与实际解码目标存在 **distribution shift**

**面试深挖点：**
```
Q: 为什么训练和推理的不一致会影响推荐效果？
A: 
1. Exposure Bias: 训练时模型只看到正确前缀，推理时会遇到错误累积
2. Loss-Evaluation Mismatch: 训练用cross-entropy，评估用HR/NDCG等排序指标
3. Beam Search 特性：beam search会倾向选择概率乘积最大的序列，
   但标准CE loss没有考虑这一点
```

---

### 1.2 三种损失函数对比

#### (1) LML (Language Model Loss)
```python
# train_lml.py:88-94
pos_logits = torch.exp(shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / self.tau)
pos_loss = -torch.log(pos_logits)
neg_logits = torch.exp(shift_logits / self.tau)
neg_loss = torch.log(neg_logits.sum(dim=-1))
loss = (pos_loss + neg_loss).mean()
```

**本质：** 标准的带温度的交叉熵损失

**面试考察点：**
- 温度τ的作用是什么？（控制softmax的平滑程度，τ→0趋向one-hot，τ→∞趋向均匀分布）
- 为什么不直接用 `F.cross_entropy`？（需要理解手动实现与内置函数的区别）

#### (2) MSL (Masked Sequence Loss)
```python
# train_msl.py:111-116
pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
pos_loss = -pos_logits / self.tau
neg_logits = torch.exp(shift_logits / self.tau)
neg_loss = torch.log(neg_logits.sum(dim=-1))
loss = (pos_loss + neg_loss).mean()
```

**关键区别：** 使用 `constrain_mask` 将无效token的logits设为 `-inf`

**面试深挖点：**
```
Q: constrain_mask 是如何生成的？为什么要用Trie？
A: 
1. 使用Trie存储所有item的token序列
2. 对于每个训练样本，根据当前已生成的token序列，
   在Trie中查找所有可能的下一个有效token
3. 只有这些有效token对应的位置mask为True
4. 这样训练时就模拟了推理时的约束解码
```

#### (3) BEAR (本文核心贡献)
```python
# train_bear.py:98-111
probs = F.softmax(logits / self.tau, dim=-1)
pos_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
ce_loss = -pos_probs.log()

# Top-K reweighting
quantile = probs.topk(self.K, dim=-1).values[:, -1].detach()
min_probs, _ = probs.masked_fill(~constrain_mask, float("inf")).min(dim=-1)
quantile = torch.where(quantile == 0, min_probs, quantile).detach()

# 关键公式：基于log概率差异的sigmoid权重
weights = torch.sigmoid((pos_probs.log() - quantile.log()) / self.topk_tau)
weights = weights.detach() * self.topk_weight + 1
```

**核心创新点：**
1. **Top-K Reweighting:** 对位于Top-K边界附近的token给予更高权重
2. **Sigmoid Gating:** 使用 `(pos_prob - quantile)` 的差值来平滑权重
3. **训练稳定性:** 权重通过 `.detach()` 阻断梯度

**面试深挖问题：**
```
Q1: 为什么用 sigmoid 而不是直接用 indicator function？
A: 
- Indicator function (I(p > q)) 在边界处不连续，梯度为0
- Sigmoid 提供平滑过渡，让梯度能够传播
- 数学上：sigmoid((log_p - log_q)/τ) ≈ σ((p-q)/τ) 在τ小时近似step function

Q2: topk_tau 和 topk_weight 的物理意义？
A:
- topk_tau (ξ): 控制sigmoid的陡峭程度
  - τ→0: 接近hard threshold，只有Top-K内的token被加权
  - τ→∞: 所有token权重趋于相同
- topk_weight (λ): 控制加权的强度
  - λ=0: 退化为标准CE loss
  - λ大: 强烈强调Top-K边界附近的token

Q3: 为什么quantile要detach？weights也要detach？
A:
- quantile.detach(): 防止梯度通过quantile回传，保持quantile作为固定参考点
- weights.detach(): 让权重成为固定的系数，不参与反向传播，稳定训练
```

---

### 1.3 Trie数据结构与约束解码

```python
# Trie.py
class Trie:
    def insert(self, token_list):
        node = self.root
        for token in token_list:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True
    
    def valid_tokens(self, token_list):
        valid_tokens_list = [list(self.root.children.keys())]
        node = self.root
        for token in token_list:
            if token in node.children:
                node = node.children[token]
                valid_tokens_list.append(list(node.children.keys()))
            else:
                valid_tokens_list.append([])
        return valid_tokens_list
```

**面试考察点：**
```
Q: Trie的时间复杂度是什么？
A: 
- 插入: O(L)，L为序列长度
- 查询: O(L)
- 空间: O(N*L*V)，N为序列数，V为平均分支因子

Q: 为什么推理时用 MarisaTrie 而训练时用自定义Trie？
A:
- MarisaTrie 是基于 marisa-trie 库的内存优化实现
- 训练时需要频繁构建和查询，自定义实现更灵活
- 推理时只需构建一次，MarisaTrie的内存效率更重要
```

---

### 1.4 推理阶段：Constrained Beam Search

```python
# inference.py 关键代码
def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list:
    input_ids = input_ids.tolist()
    for i in range(len(input_ids)):
        if input_ids[i : i + len(sep)] == sep:
            break
    prefix = input_ids[i:]
    allowed_tokens = trie.get(prefix)
    allowed_tokens = [tokenizer.eos_token_id] if allowed_tokens == [] else allowed_tokens
    return allowed_tokens
```

**面试深挖点：**
```
Q: CBS (Constrained Beam Search) 和普通Beam Search的区别？
A:
1. 普通BS: 每一步从整个vocab中选top-k
2. CBS: 每一步只从允许的token集合中选top-k
3. CBS保证生成的序列一定是有效的item title

Q: customCBS_model.py 中的修改是什么？
A:
修改了 logits_processor 的应用顺序：
- 原版: logits → log_softmax → logits_processor → 加beam_scores
- 修改: logits → logits_processor → log_softmax → 加beam_scores
这样约束在softmax之前应用，确保被mask的token概率严格为0
```

---

## 第二部分：LLM核心知识点

### 2.1 Transformer架构

**必考问题：**
```
Q1: Self-Attention的计算复杂度是多少？如何优化？
A: O(n²d)，优化方法：
- Flash Attention: IO-aware，减少HBM访问
- Sparse Attention: 局部/全局模式
- Linear Attention: 核近似

Q2: 为什么用RoPE而不是绝对位置编码？
A:
- 绝对位置编码：泛化到更长序列能力差
- RoPE: 相对位置信息，支持长度外推
- 数学上：q·k 包含相对位置信息 m-n

Q3: LayerNorm vs RMSNorm的区别？
A:
- LayerNorm: 减均值除标准差，有两个参数(γ,β)
- RMSNorm: 只除RMS，省略re-centering，只有一个参数γ
- RMSNorm更快且效果相当，LLaMA系列使用
```

### 2.2 训练技术

#### LoRA (Low-Rank Adaptation)
```python
# 项目中的LoRA配置
config = LoraConfig(
    r=lora_r,           # rank，通常8-64
    lora_alpha=lora_alpha,  # scaling factor
    target_modules=lora_target_modules,  # ["q_proj", "v_proj"]
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**面试必考：**
```
Q1: LoRA的数学原理？
A: W = W₀ + BA，其中 B∈R^(d×r), A∈R^(r×k), r << min(d,k)
- 冻结W₀，只训练A和B
- 推理时可以合并：W = W₀ + BA，无额外推理开销

Q2: 为什么选择q_proj和v_proj？
A: 
- 经验发现attention层的参数更重要
- q_proj影响"关注什么"，v_proj影响"传递什么信息"
- 也可以选择所有线性层，但成本更高

Q3: lora_alpha的作用？
A: scaling = alpha / r
- 控制LoRA更新的幅度
- alpha=r时，scaling=1
- 通常alpha=2r，即scaling=2
```

#### 量化训练
```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
```

**面试考察：**
```
Q: 8-bit量化的原理？
A: 
- LLM.int8(): 混合精度，outlier用FP16，其余用INT8
- Outlier: 幅度>阈值(通常6.0)的激活值
- 通过absmax量化：X_int8 = round(X / scale * 127)
```

### 2.3 分布式训练

```python
# 使用Accelerate
accelerator = Accelerator()
world_size = int(os.environ.get("WORLD_SIZE", 1))
micro_batch_size = batch_size // world_size
gradient_accumulation_steps = batch_size // micro_batch_size // world_size
```

**面试必考：**
```
Q1: 数据并行 vs 模型并行 vs 流水线并行？
A:
- 数据并行 (DP/DDP): 每个GPU完整模型，不同数据
- 模型并行 (Tensor Parallel): 一层拆到多个GPU
- 流水线并行 (Pipeline Parallel): 不同层放不同GPU

Q2: DDP中的梯度同步机制？
A:
- AllReduce: 所有GPU的梯度求平均
- 梯度bucket: 多个参数的梯度打包通信
- 梯度累积: 多个micro-batch累积后同步

Q3: ZeRO优化的三个阶段？
A:
- Stage 1: 优化器状态分片
- Stage 2: 梯度分片
- Stage 3: 参数分片
```

---

## 第三部分：生成式推荐(GR)核心问题

### 3.1 生成式推荐 vs 传统推荐

**面试框架：**
```
Q: 生成式推荐相比传统协同过滤/双塔模型的优势？

传统方法：
- 双塔模型: user塔 + item塔，ANN检索
- 局限: 无法建模复杂序列依赖，冷启动问题

生成式推荐：
1. 统一建模: 将推荐转化为序列生成问题
2. 知识迁移: LLM的world knowledge
3. 零样本能力: 可以处理新item
4. 自然语言交互: 可解释性

局限性：
1. 推理效率: 逐token生成，无法并行
2. 幻觉问题: 可能生成不存在的item
3. 位置偏差: 自回归模型有位置偏好
```

### 3.2 约束解码在推荐中的应用

**面试深挖：**
```
Q: 为什么推荐系统需要约束解码？

问题：
1. LLM可能生成不在item库中的标题（幻觉）
2. 生成的格式可能不符合要求（缺少引号等）
3. 相同item可能有多种表述方式

解决方案：
1. Trie约束: 只允许生成有效的item序列
2. 前缀约束: 确保格式一致性
3. 词表约束: 限制在有效token范围内

Q: 约束解码的trade-off？
A:
- 优点: 保证输出有效性，提高准确率
- 缺点: 
  - 推理速度变慢（需要查询Trie）
  - 可能过于约束，限制模型表达能力
  - 训练-推理不一致（BEAR就是解决这个问题）
```

### 3.3 推荐评估指标

```python
# evaluate_batch_match.py
def evaluate(rank_list, topk_list=[1, 5, 10]):
    rank_array = np.array(rank_list)
    NDCG, HR = [], []
    for k in topk_list:
        Hit_num = (rank_array < k).sum()
        HR.append(Hit_num / len(rank_array))
        mask = rank_array < k
        NDCG_num = 1 / np.log2(rank_array[mask] + 2)
        NDCG.append(NDCG_num.sum() / len(rank_array) / (1.0 / math.log2(2)))
```

**面试考察：**
```
Q1: HR@K 和 NDCG@K 的区别？
A:
- HR@K: 命中率，只要ground truth在top-K中就算1
- NDCG@K: 考虑位置，越靠前得分越高
- NDCG = DCG/IDCG，DCG = Σ(1/log2(i+1))

Q2: 为什么用 rank 而不是直接用分数？
A:
- 生成式模型输出的是序列，不是分数
- rank是根据beam search的序列分数排序得到的
- 与实际使用场景一致（用户看到的是排序后的列表）
```

---

## 第四部分：Agent相关考点

### 4.1 LLM Agent基础架构

**面试框架：**
```
Q: 描述一个典型的LLM Agent系统架构？

核心组件：
1. Planning: 任务分解、子目标设定
2. Memory: 
   - 短期记忆: 上下文窗口
   - 长期记忆: 向量数据库/外部存储
3. Tools: API调用、代码执行、搜索引擎
4. Reflection: 自我评估、错误修正

工作流程：
Observation → Thought → Action → Observation → ...
```

### 4.2 ReAct模式

**面试深挖：**
```
Q: ReAct (Reasoning + Acting) 的工作原理？

Thought: 推理当前状态，规划下一步
Action: 执行具体操作（搜索/计算/API调用）
Observation: 获取操作结果
... 循环直到完成任务

示例：
用户: "推荐一本类似《三体》的科幻小说"
Thought: 用户喜欢硬科幻，需要搜索类似作品
Action: search("硬科幻小说 推荐 类似三体")
Observation: 《基地》《沙丘》《2001太空漫游》...
Thought: 需要根据用户偏好筛选
Answer: "推荐《基地》系列，阿西莫夫的硬科幻经典..."
```

### 4.3 Tool Use与Function Calling

**面试考察：**
```
Q1: Function Calling的实现原理？
A:
1. 定义工具schema（名称、参数、描述）
2. 在prompt中注入工具信息
3. 模型输出特殊格式的function call
4. 系统执行函数，返回结果
5. 模型基于结果继续生成

Q2: 如何设计好的工具描述？
A:
- 清晰的参数说明和示例
- 明确的使用场景
- 错误处理指引
- 避免歧义
```

### 4.4 Agent在推荐系统中的应用

**面试深挖：**
```
Q: 如何用Agent增强推荐系统？

1. 多轮对话推荐：
   - Clarification: 询问用户偏好
   - Explanation: 解释推荐理由
   - Refinement: 根据反馈调整

2. 工具增强推荐：
   - 搜索引擎: 获取实时信息
   - 知识图谱: 获取item属性
   - 用户画像: 理解用户历史

3. 自适应推荐策略：
   - 根据用户类型选择推荐算法
   - 根据场景调整推荐多样性
   - 根据反馈动态更新
```

---

## 第五部分：后训练(Post-training)技术

### 5.1 SFT (Supervised Fine-Tuning)

**面试必考：**
```
Q1: SFT的数据格式？
A: 
{
  "instruction": "任务描述",
  "input": "输入内容",
  "output": "期望输出"
}

Q2: SFT的关键超参数？
A:
- Learning Rate: 通常1e-5 ~ 5e-5
- Epochs: 1-3个epoch，避免过拟合
- Batch Size: 根据显存调整
- Warmup: 防止初期梯度爆炸

Q3: 如何判断SFT是否过拟合？
A:
- 训练loss下降但eval loss上升
- 生成质量变差（重复、退化）
- 在held-out数据上性能下降
```

### 5.2 RLHF (Reinforcement Learning from Human Feedback)

**面试框架：**
```
Q: 描述RLHF的完整流程？

阶段1: SFT
- 用高质量数据微调基础模型

阶段2: Reward Model Training
- 收集人类偏好数据（A>B>C）
- 训练reward model: RM(prompt, response) → score

阶段3: PPO优化
- 用reward model作为奖励信号
- PPO优化策略模型
- KL散度约束防止偏离太远

关键公式：
J(θ) = E[R(x,y)] - β * KL(π_θ || π_ref)
```

### 5.3 DPO (Direct Preference Optimization)

**面试深挖：**
```
Q1: DPO相比RLHF的优势？
A:
- 无需训练reward model
- 无需PPO的复杂实现
- 直接从偏好数据学习
- 更稳定，更易调参

Q2: DPO的损失函数？
A:
L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中：
- y_w: 偏好响应
- y_l: 非偏好响应
- π_ref: 参考策略（SFT后的模型）
- β: 温度参数

Q3: DPO的局限性？
A:
- 需要成对偏好数据
- 可能存在reward hacking
- 对数据质量敏感
- 离线方法，无法在线探索
```

### 5.4 本项目中的后训练技术

**项目使用的技术栈：**
```
1. LoRA微调: 参数高效
2. 8-bit量化: 显存优化
3. 自定义损失: 针对推荐任务优化
4. 约束解码: 保证输出有效性

面试可考察：
Q: 为什么这个项目用LoRA而不是全参数微调？
A:
- 推荐数据通常较小，全参数易过拟合
- LoRA显存效率高，可以跑更大batch
- 推理时合并，无额外开销
- 多个推荐场景可以共享基础模型
```

---

## 第六部分：高频面试深挖问题

### 6.1 代码理解类

```
Q1: 解释 train_bear.py 中 compute_loss 的完整流程？

A:
1. 提取constrain_mask，获取labels和logits
2. 对齐维度：只取输出部分的logits
3. 应用约束：logits[~constrain_mask] = -inf
4. 计算概率：probs = softmax(logits/τ)
5. 获取正样本概率：pos_probs = gather(probs, labels)
6. 计算CE损失：ce_loss = -log(pos_probs)
7. 计算Top-K分位数：quantile = topk(probs, K)[:,-1]
8. 计算权重：sigmoid((log(pos) - log(quantile))/ξ)
9. 归一化权重
10. 加权求和得到最终损失

Q2: 为什么需要 CustomDataCollatorForSeq2Seq？
A:
- 标准collator处理不了constrain_mask（3D tensor）
- 需要单独pad constrain_mask并保持对齐
- padding_value=1表示默认所有token都有效
```

### 6.2 设计思想类

```
Q1: 为什么BEAR选择在训练时模拟推理的约束？
A:
- 核心思想：训练-推理一致性
- 如果训练时不约束，模型会学习到"错误"的概率分布
- 推理时约束会改变分布，导致次优解
- 提前让模型"知道"约束，学习更准确的分布

Q2: Top-K Reweighting的直觉解释？
A:
- 想象beam search在Top-K边界做决策
- 边界附近的token概率很接近，但选择不同
- 标准CE对所有token一视同仁
- BEAR强调这些"临界点"，让模型更好地区分它们

Q3: 为什么用sigmoid而不是直接加权？
A:
- 直接加权（如I(p>q)）在边界处不连续
- 梯度为0，无法学习
- Sigmoid提供平滑过渡，梯度可以传播
- 数学上更优雅，实践中更稳定
```

### 6.3 扩展思考类

```
Q1: 如果要将BEAR应用到其他生成任务（如文本生成），需要改什么？
A:
1. 重新定义"有效token"：不再是item库，而是语法规则/词表
2. 修改Trie构建方式：根据任务定义约束
3. 调整Top-K：根据任务难度调整K值
4. 可能需要不同的评估指标

Q2: BEAR和RLHF/DPO的关系？
A:
- 都是优化生成质量
- RLHF/DPO: 通过偏好数据优化
- BEAR: 通过任务特定约束优化
- 可以结合：先BEAR训练，再RLHF微调

Q3: 如何处理动态item库（新item不断加入）？
A:
- 增量更新Trie
- 定期重训模型
- 考虑用检索+生成的混合方案
- 或者学习item embedding而非title生成
```

---

## 第七部分：系统设计类问题

### 7.1 设计一个LLM推荐系统

**面试框架：**
```
需求：设计一个基于LLM的商品推荐系统

1. 数据层：
   - 用户行为数据（点击、购买、评分）
   - Item属性数据（标题、描述、类别）
   - 上下文数据（时间、地点、设备）

2. 模型层：
   - 基础LLM（如LLaMA）
   - LoRA适配器（针对推荐任务）
   - 约束解码模块（Trie）

3. 服务层：
   - 推理服务（vLLM/TGI）
   - 缓存层（用户画像、热门item）
   - 监控和日志

4. 评估层：
   - 离线指标（HR@K, NDCG@K）
   - 在线指标（CTR, 转化率）
   - A/B测试框架
```

### 7.2 优化推理效率

**面试深挖：**
```
Q: 如何优化生成式推荐的推理速度？

1. 模型层面：
   - 量化（INT8/INT4）
   - 模型蒸馏
   - 剪枝

2. 推理层面：
   - KV Cache
   - 批量推理
   - 投机解码（Speculative Decoding）

3. 系统层面：
   - 模型并行
   - 动态批处理
   - 请求调度

4. 缓存层面：
   - 热门用户结果缓存
   - Embedding缓存
   - 历史上下文缓存
```

---

## 第八部分：论文阅读与复现

### 8.1 如何阅读本论文

**面试可展示的能力：**
```
1. 问题定义：训练-推理不一致
2. 动机分析：现有方法的局限
3. 方法设计：BEAR的核心创新
4. 理论分析：为什么有效
5. 实验验证：消融实验、对比实验
6. 代码复现：理解实现细节
```

### 8.2 复现时遇到的挑战

**面试可分享的点：**
```
1. 数据处理：
   - Item ID到Title的映射
   - 序列构建和划分
   - 特殊字符处理（&amp;等）

2. 训练技巧：
   - 梯度累积的正确实现
   - 分布式训练的同步问题
   - 显存优化（8-bit + gradient checkpointing）

3. 评估挑战：
   - 离线评估与在线的gap
   - 评估指标的选择
   - 结果的可复现性
```

---

## 第九部分：行为面试与项目经验

### 9.1 如何介绍这个项目

**STAR法则：**
```
Situation: 
LLM用于推荐系统，但训练和推理存在不一致

Task:
设计一种训练方法，使模型在训练时就考虑推理的约束

Action:
1. 分析问题：训练用teacher forcing，推理用beam search
2. 提出方案：BEAR，训练时加入约束和Top-K重加权
3. 实现细节：Trie约束、sigmoid平滑、权重归一化
4. 实验验证：在多个数据集上验证有效性

Result:
- HR@10提升X%，NDCG@10提升Y%
- 训练稳定性提高
- 推理效率无额外开销
```

### 9.2 如何回答"你学到了什么"

```
技术层面：
1. LLM微调的最佳实践（LoRA、量化、分布式）
2. 训练-推理一致性的重要性
3. 约束解码的实现原理
4. 推荐系统的评估方法

方法论层面：
1. 问题驱动的研究方法
2. 从简单baseline到复杂方法的演进
3. 消融实验的重要性
4. 代码复现的工程能力
```

---

## 第十部分：常见陷阱与应对

### 10.1 不会的问题

**应对策略：**
```
1. 诚实承认："这个我没有深入研究过"
2. 展示思路："但我可以从XX角度分析"
3. 关联已知："这和我了解的XX有些相似"
4. 请教态度："您能给个提示吗？"
```

### 10.2 代码题应对

**准备方向：**
```
1. 手写Attention
2. 实现LoRA
3. 写Beam Search
4. 实现Trie
5. 计算NDCG
```

---

## 附录：关键代码片段

### A.1 核心损失函数对比

```python
# LML (标准CE)
loss = -log(softmax(logits/τ)[label])

# MSL (带约束的CE)
logits[~mask] = -inf
loss = -log(softmax(logits/τ)[label])

# BEAR (Top-K重加权CE)
probs = softmax(logits/τ)
quantile = topk(probs, K)[:,-1]
weight = sigmoid((log(probs[label]) - log(quantile))/ξ) * λ + 1
loss = -log(probs[label]) * weight
```

### A.2 Trie约束解码流程

```python
# 训练时
trie = Trie()
for item_title in all_items:
    tokens = tokenizer.encode(item_title)
    trie.insert(tokens)

# 对于每个训练样本
valid_tokens = trie.valid_tokens(current_tokens)
constrain_mask = create_mask(valid_tokens)
logits[~constrain_mask] = -inf

# 推理时
def prefix_allowed_tokens_fn(batch_id, input_ids):
    prefix = find_prefix_after_sep(input_ids)
    allowed = trie.get(prefix)
    return allowed or [eos_token_id]
```

### A.3 分布式训练关键配置

```python
# accelerate配置
accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision="bf16",
    log_with="tensorboard",
)

# 训练循环
with accelerator.accumulate(model):
    outputs = model(**batch)
    loss = compute_loss(outputs)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 总结：面试准备Checklist

- [ ] 理解BEAR的核心创新点
- [ ] 能解释三种损失的区别
- [ ] 理解Trie约束解码的原理
- [ ] 掌握LoRA的数学原理和实现
- [ ] 了解分布式训练的基本概念
- [ ] 能设计一个LLM推荐系统
- [ ] 熟悉RLHF/DPO等后训练技术
- [ ] 准备好项目介绍的STAR故事
- [ ] 能手写核心代码片段
- [ ] 了解Agent的基本架构

---

*最后更新：2026年6月*
*基于 efficient-BEAR 项目*
