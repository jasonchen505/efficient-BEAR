# efficient-BEAR 复现计划

> 硬件资源：8 x NVIDIA RTX 3090 (24GB显存)
> 目标：完整复现BEAR论文的核心实验

---

## 一、资源评估与可行性分析

### 1.1 硬件资源对比

| 项目 | 论文配置 | 我们的配置 | 影响 |
|------|----------|------------|------|
| GPU | A100 (80GB) | RTX 3090 (24GB) | 需要减小batch或用梯度累积 |
| 精度 | bf16 | fp16 | 3090不原生支持bf16，用fp16替代 |
| 卡数 | 8 | 8 | 一致 |
| 显存总计 | 640GB | 192GB | 足够跑3B模型 |

### 1.2 模型显存估算

```
LLaMA-3.2-3B 显存估算：
- FP16: ~6GB
- INT8: ~3GB
- LoRA参数(r=8): ~50MB
- 优化器状态: ~200MB
- 激活值(batch=16): ~4GB
- 总计: ~8-10GB/卡 (INT8 + LoRA)

结论：3090的24GB显存完全够用，甚至可以用更大的batch
```

### 1.3 关键兼容性问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| bf16不支持 | 3090不原生支持bf16 | 改用fp16，或用tf32 |
| PyTorch版本 | 原项目用nightly版本 | 用稳定版PyTorch 2.x |
| genre库 | 可能有版本冲突 | 从源码安装或找替代 |
| flash-attention | 3090支持但需要编译 | 安装预编译版本 |

---

## 二、环境配置计划

### 2.1 基础环境

```bash
# 创建conda环境
conda create -n bear python=3.11 -y
conda activate bear

# 安装PyTorch (稳定版，支持3090)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install transformers==4.49.0
pip install peft==0.15.2
pip install accelerate==1.8.1
pip install bitsandbytes==0.46.0
pip install datasets==3.6.0
pip install fire==0.7.0
pip install pandas==2.3.0
pip install tensorboard==2.19.0

# 安装genre库 (用于推理时的MarisaTrie)
pip install genre==0.1.3
# 如果失败，从源码安装：
# pip install git+https://github.com/facebookresearch/GENRE.git

# 安装marisa-trie
pip install marisa-trie==1.2.1
```

### 2.2 修改代码适配3090

需要修改的关键文件：

**1. train_bear.py / train_lml.py / train_msl.py**
```python
# 修改前 (使用bf16)
args=transformers.TrainingArguments(
    ...
    bf16=True,
    tf32=True,
    ...
)

# 修改后 (使用fp16)
args=transformers.TrainingArguments(
    ...
    fp16=True,  # 改用fp16
    tf32=True,
    ...
)
```

**2. 模型加载部分**
```python
# 修改前
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,  # bf16
    ...
)

# 修改后
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,  # fp16
    ...
)
```

**3. inference.py**
```python
# 同样修改torch_dtype
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,  # 改为fp16
    ...
)
```

---

## 三、数据准备计划

### 3.1 现有数据

项目自带4个数据集：
```
data/
├── Book/
├── Clothing/
├── Office/
└── Toy/
```

每个数据集包含：
- `id2name4Rec.json`: item ID到名称的映射
- `train_10000.csv`: 训练数据
- `test_5000.csv`: 测试数据

### 3.2 数据格式理解

```csv
user_id,item_ids,mask,item_titles,replaced_item_titles,timestamp
1431,"[8383, 8397, 6917, ...]","[0, 0, 0, ...]","[""title1"", ...]",[],1360886400
```

关键字段：
- `user_id`: 用户ID
- `item_ids`: 用户交互过的item序列
- `item_titles`: 对应的item标题

### 3.3 数据预处理流程

```python
# 1. 加载id2title映射
id2title_dict = {int(k): v for k, v in json.load(open("id2name4Rec.json")).items()}

# 2. 构建训练样本
# 输入: 用户历史 ("item1", "item2", ...)
# 输出: 下一个item ("item3")

# 3. 构建prompt
instruction = "Given a list of toys the user has played before, please recommend a new toy..."
input_str = "The user has played the following toys before: \"item1\", \"item2\", ..."
output_str = "\"item3\""
```

---

## 四、训练计划

### 4.1 阶段一：单卡调试 (Day 1)

**目标**：确保代码能在单卡3090上跑通

```bash
# 单卡训练命令
CUDA_VISIBLE_DEVICES=0 python train_bear.py \
    --dataset_name Toy \
    --base_model /path/to/Llama-3.2-3B \
    --sample 1000 \
    --batch_size 8 \
    --num_epochs 2 \
    --tau 1.0 \
    --topk_tau 2.75 \
    --topk_weight 0.25
```

**检查点**：
- [ ] 模型加载成功
- [ ] 数据处理成功
- [ ] 训练loss正常下降
- [ ] 显存占用合理 (<20GB)

### 4.2 阶段二：多卡训练 (Day 2-3)

**目标**：用8卡完成完整训练

```bash
# 8卡训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    train_bear.py \
    --dataset_name Toy \
    --base_model /path/to/Llama-3.2-3B \
    --sample 10000 \
    --batch_size 128 \
    --num_epochs 10 \
    --tau 1.0 \
    --topk_tau 2.75 \
    --topk_weight 0.25
```

**超参数设置**：
| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 128 | 全局batch size |
| micro_batch | 16 | 每卡batch size (128/8) |
| gradient_accumulation | 1 | 无需累积 |
| learning_rate | 1e-4 | AdamW优化器 |
| warmup_steps | 20 | 预热步数 |
| num_epochs | 10 | 训练轮数 |
| lora_r | 8 | LoRA rank |
| lora_alpha | 16 | LoRA scaling |

### 4.3 阶段三：三种方法对比 (Day 3-5)

**目标**：复现LML、MSL、BEAR的对比实验

```bash
# 1. LML (Baseline)
python cmd/run.py Toy --loss lml --model_path /path/to/Llama-3.2-3B \
    --cuda 0,1,2,3,4,5,6,7 --num_epochs 10

# 2. MSL (带约束)
python cmd/run.py Toy --loss msl --model_path /path/to/Llama-3.2-3B \
    --cuda 0,1,2,3,4,5,6,7 --num_epochs 10

# 3. BEAR (本文方法)
python cmd/run.py Toy --loss bear --model_path /path/to/Llama-3.2-3B \
    --cuda 0,1,2,3,4,5,6,7 --num_epochs 10 \
    --tau 1.0 --topk_tau 2.75 --topk_weight 0.25
```

### 4.4 阶段四：消融实验 (Day 5-7)

**目标**：验证BEAR各组件的有效性

**实验1: Top-K值的影响**
```bash
for K in 1 5 10 20 50; do
    python cmd/run.py Toy --loss bear --model_path /path/to/Llama-3.2-3B \
        --cuda 0,1,2,3,4,5,6,7 --num_epochs 10 \
        --tau 1.0 --topk_tau 2.75 --topk_weight 0.25
done
```

**实验2: 温度τ的影响**
```bash
for tau in 0.5 1.0 2.0 5.0; do
    python cmd/run.py Toy --loss bear --model_path /path/to/Llama-3.2-3B \
        --cuda 0,1,2,3,4,5,6,7 --num_epochs 10 \
        --tau $tau --topk_tau 2.75 --topk_weight 0.25
done
```

**实验3: 权重λ的影响**
```bash
for lambda in 0.0 0.1 0.25 0.5 1.0; do
    python cmd/run.py Toy --loss bear --model_path /path/to/Llama-3.2-3B \
        --cuda 0,1,2,3,4,5,6,7 --num_epochs 10 \
        --tau 1.0 --topk_tau 2.75 --topk_weight $lambda
done
```

---

## 五、评估计划

### 5.1 评估指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| HR@1 | 命中率Top-1 | 第一个推荐是否正确 |
| HR@5 | 命中率Top-5 | 前5个推荐是否包含正确答案 |
| HR@10 | 命中率Top-10 | 前10个推荐是否包含正确答案 |
| NDCG@1 | 归一化折损累积增益Top-1 | 考虑位置的准确率 |
| NDCG@5 | 归一化折损累积增益Top-5 | 考虑位置的准确率 |
| NDCG@10 | 归一化折损累积增益Top-10 | 考虑位置的准确率 |

### 5.2 评估流程

```bash
# 1. 推理生成预测结果
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    inference.py \
    --base_model /path/to/Llama-3.2-3B \
    --lora_weights_path /path/to/checkpoint \
    --dataset Toy \
    --constrained_before_softmax 1

# 2. 计算评估指标
python evaluate_batch_match.py \
    --lora_weights_father_path /path/to/save_dir

# 3. 汇总结果
python read_metric.py /path/to/save_dir customCBS
```

### 5.3 预期结果

根据论文，预期结果范围：

| 方法 | HR@10 | NDCG@10 |
|------|-------|---------|
| LML | ~0.30 | ~0.15 |
| MSL | ~0.35 | ~0.18 |
| BEAR | ~0.40 | ~0.22 |

**注意**：实际结果可能因随机种子、数据划分等因素有差异

---

## 六、时间规划

### 6.1 总体时间表

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| Day 1 | 环境配置 + 单卡调试 | 环境搭建、代码修改、单卡测试 | 可运行的代码 |
| Day 2-3 | 多卡训练 | 8卡训练LML/MSL/BEAR | 三个模型的checkpoint |
| Day 4 | 评估 | 推理+评估所有checkpoint | 评估结果表格 |
| Day 5-6 | 消融实验 | 超参数敏感性实验 | 消融实验结果 |
| Day 7 | 总结 | 结果分析、写报告 | 复现报告 |

### 6.2 每日详细计划

**Day 1: 环境配置与单卡调试**
```
上午 (4小时):
- [ ] 安装conda环境
- [ ] 下载LLaMA-3.2-3B模型
- [ ] 安装所有依赖库
- [ ] 解决兼容性问题

下午 (4小时):
- [ ] 修改代码适配3090 (bf16 -> fp16)
- [ ] 单卡运行train_bear.py (sample=1000)
- [ ] 检查训练loss是否正常
- [ ] 检查显存占用
```

**Day 2: 多卡训练 - LML和MSL**
```
上午 (4小时):
- [ ] 配置accelerate多卡设置
- [ ] 运行LML训练 (sample=10000, epoch=10)
- [ ] 运行MSL训练 (sample=10000, epoch=10)

下午 (4小时):
- [ ] 监控训练过程
- [ ] 检查checkpoint保存
- [ ] 准备BEAR训练
```

**Day 3: 多卡训练 - BEAR**
```
全天 (8小时):
- [ ] 运行BEAR训练 (sample=10000, epoch=10)
- [ ] 监控训练过程
- [ ] 保存所有checkpoint
```

**Day 4: 评估**
```
上午 (4小时):
- [ ] 对所有checkpoint运行推理
- [ ] 生成预测结果文件

下午 (4小时):
- [ ] 计算评估指标
- [ ] 汇总结果到表格
- [ ] 分析结果
```

**Day 5-6: 消融实验**
```
Day 5:
- [ ] Top-K值消融实验
- [ ] 温度τ消融实验

Day 6:
- [ ] 权重λ消融实验
- [ ] 汇总消融实验结果
```

**Day 7: 总结**
```
全天:
- [ ] 整理所有实验结果
- [ ] 分析结果并得出结论
- [ ] 撰写复现报告
- [ ] 准备presentation
```

---

## 七、风险与应对

### 7.1 潜在风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| 环境安装失败 | 中 | 高 | 提前准备Docker镜像 |
| 显存不足 | 低 | 中 | 减小batch_size，增加梯度累积 |
| 训练loss不收敛 | 低 | 高 | 检查数据、调整学习率 |
| 评估结果与论文差异大 | 中 | 中 | 检查数据划分、随机种子 |
| 时间不够 | 中 | 中 | 优先完成核心实验 |

### 7.2 备选方案

**方案A: 如果显存不足**
```python
# 减小batch_size，增加梯度累积
batch_size = 64  # 原来128
gradient_accumulation_steps = 2  # 原来1
```

**方案B: 如果训练太慢**
```python
# 减少训练数据
sample = 5000  # 原来10000

# 减少epoch
num_epochs = 5  # 原来10
```

**方案C: 如果genre库安装失败**
```python
# 使用自定义Trie替代MarisaTrie
# 训练时已经用的是自定义Trie，不受影响
# 推理时修改inference.py，用自定义Trie
```

---

## 八、关键代码修改清单

### 8.1 必须修改的文件

| 文件 | 修改内容 | 原因 |
|------|----------|------|
| train_bear.py | bf16=True -> fp16=True | 3090不支持bf16 |
| train_lml.py | bf16=True -> fp16=True | 同上 |
| train_msl.py | bf16=True -> fp16=True | 同上 |
| inference.py | torch_dtype=torch.bfloat16 -> float16 | 同上 |
| cmd/run.py | 修改code_dir路径 | 适配我们的环境 |

### 8.2 可选优化

| 文件 | 优化内容 | 好处 |
|------|----------|------|
| train_bear.py | 添加gradient_checkpointing | 减少显存占用 |
| inference.py | 添加batch推理 | 提高推理速度 |
| 所有训练文件 | 添加更多logging | 方便调试 |

### 8.3 具体修改示例

**train_bear.py 修改：**
```python
# 第305-306行
# 修改前
bf16=True,
tf32=True,

# 修改后
fp16=True,
tf32=True,
```

**inference.py 修改：**
```python
# 第97-101行
# 修改前
model = CustomLlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    ...
)

# 修改后
model = CustomLlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    ...
)
```

---

## 九、验证checklist

### 9.1 环境验证
- [ ] Python版本正确 (3.11)
- [ ] PyTorch版本正确 (2.1.x)
- [ ] CUDA可用 (nvidia-smi)
- [ ] 所有依赖安装成功

### 9.2 数据验证
- [ ] 数据文件存在
- [ ] 数据格式正确
- [ ] id2title映射完整

### 9.3 训练验证
- [ ] 模型加载成功
- [ ] LoRA配置正确
- [ ] 训练loss正常下降
- [ ] 显存占用合理
- [ ] Checkpoint保存正常

### 9.4 评估验证
- [ ] 推理生成成功
- [ ] 评估指标计算正确
- [ ] 结果文件保存正常

---

## 十、学习目标

### 10.1 技术理解
- [ ] 理解BEAR的核心创新
- [ ] 理解三种损失函数的区别
- [ ] 理解Trie约束解码的原理
- [ ] 理解训练-推理一致性的重要性

### 10.2 工程能力
- [ ] 掌握LoRA微调的实现
- [ ] 掌握分布式训练的配置
- [ ] 掌握约束解码的实现
- [ ] 掌握评估流程的设计

### 10.3 实验能力
- [ ] 能设计消融实验
- [ ] 能分析实验结果
- [ ] 能定位和解决问题
- [ ] 能撰写实验报告

---

*最后更新：2026年6月*
*硬件环境：8 x NVIDIA RTX 3090*
