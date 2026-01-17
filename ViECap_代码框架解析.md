# ViECap 代码框架解析与运行指南

## 一、项目概述

**ViECap** (Transferable Decoding with Visual Entities for Zero-Shot Image Captioning) 是一个零样本图像描述生成模型，发表于 ICCV 2023。

### 核心特点
- **零样本跨域迁移**：在未见过的领域也能生成高质量描述
- **实体感知解码**：使用视觉实体（Visual Entities）指导描述生成
- **双提示机制**：结合软提示（Soft Prompt）和硬提示（Hard Prompt）

## 二、代码架构

### 2.1 核心模块

#### 1. **ClipCap.py** - 模型架构
```python
# 主要组件：
- MappingNetwork: CLIP特征到语言模型的映射网络
  - Transformer投影层：将CLIP特征投影到GPT隐藏空间
  - 可学习前缀嵌入：生成软提示（soft prompts）
  
- ClipCaptionModel: 主模型
  - 输入：CLIP图像特征
  - 输出：文本描述
  - 支持：软提示 + 硬提示的组合方式
```

**关键类**：
- `MappingNetwork`: 将CLIP特征映射为连续提示嵌入
- `ClipCaptionModel`: 完整的图像描述生成模型
- `ClipCaptionPrefix`: 冻结GPT，只训练映射网络的变体

#### 2. **main.py** - 训练脚本
- 数据加载与预处理
- 模型训练循环
- 检查点保存

#### 3. **infer_by_instance.py** - 单图推理
- 加载训练好的模型
- 图像特征提取
- 实体检测与硬提示生成
- 文本生成（贪心搜索/束搜索）

#### 4. **validation.py** - 评估脚本
- COCO/Flickr30k/NoCaps数据集评估
- 批量生成描述
- 结果保存为JSON格式

#### 5. **CaptionsDataset.py** - 数据集处理
- 加载带实体的标注数据
- 实体处理与硬提示构建
- 文本tokenization与padding

#### 6. **utils.py** - 工具函数
- `noise_injection`: 噪声注入（数据增强）
- `entities_process`: 实体处理与过滤
- `compose_discrete_prompts`: 构建硬提示模板
- `padding_captions`: 序列填充

### 2.2 数据流程

```
训练阶段：
图像/文本 → CLIP编码 → 特征提取 → 实体检测 → 硬提示构建
                ↓
        映射网络（MappingNetwork）
                ↓
        软提示生成
                ↓
    [软提示 + 硬提示 + 描述] → GPT-2 → 损失计算
```

```
推理阶段：
图像 → CLIP编码 → 图像特征
                ↓
        映射网络 → 软提示
                ↓
    CLIP图像-文本相似度 → Top-K实体检测 → 硬提示
                ↓
    [软提示 + 硬提示] → GPT-2 → 文本生成
```

### 2.3 关键概念

#### 软提示（Soft Prompt）
- 由映射网络从CLIP特征生成的连续嵌入向量
- 长度：`continuous_prompt_length`（默认10）
- 位置：可在硬提示之前或之后

#### 硬提示（Hard Prompt）
- 从图像中检测到的实体构建的离散文本提示
- 格式：`"There are [entity1], [entity2], ... in image."`
- 通过CLIP图像-文本相似度检索Top-K实体

#### 实体检测
- 使用预定义的实体词汇表（Visual Genome/COCO/Open Images等）
- 计算图像特征与实体文本嵌入的相似度
- 选择Top-K个最相关的实体

## 三、环境配置

### 3.1 依赖安装

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖：
- clip (OpenAI CLIP)
- transformers==4.19.2
- torch
- tqdm
- nltk
- pycocotools
```

### 3.2 数据准备

1. **下载预训练检查点**（从Releases下载）
   - 模型权重
   - 预处理的数据集
   - 实体词汇表

2. **数据预处理**（可选，如果使用自己的数据）

```bash
# 1. 提取实体
python entities_extraction.py

# 2. 提取文本特征（可选）
python texts_features_extraction.py

# 3. 生成提示集合嵌入（用于推理）
python generating_prompt_ensemble.py

# 4. 提取图像特征（可选，用于评估）
python images_features_extraction.py
```

## 四、模型运行

### 4.1 训练模型

#### COCO数据集训练
```bash
bash scripts/train_coco.sh 0
# 参数：0 表示使用 cuda:0
```

**训练参数**（在`scripts/train_coco.sh`中）：
- `--bs 80`: 批次大小
- `--lr 0.00002`: 学习率
- `--epochs 15`: 训练轮数
- `--device cuda:0`: GPU设备
- `--using_hard_prompt`: 使用硬提示
- `--soft_prompt_first`: 软提示在前
- `--clip_model ViT-B/32`: CLIP模型类型

#### Flickr30k数据集训练
```bash
bash scripts/train_flickr30k.sh 0
```

#### 自定义训练
```bash
python main.py \
    --bs 80 \
    --lr 0.00002 \
    --epochs 15 \
    --device cuda:0 \
    --using_hard_prompt \
    --soft_prompt_first \
    --path_of_datasets ./annotations/coco/coco_texts_features_ViT-B32.pickle \
    --out_dir ./checkpoints/train_coco \
    --prefix coco_prefix
```

### 4.2 单图推理

```bash
python infer_by_instance.py \
    --prompt_ensemble \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt \
    --device cuda:0 \
    --clip_model ViT-B/32 \
    --top_k 3 \
    --threshold 0.2
```

**关键参数**：
- `--image_path`: 图像路径
- `--weight_path`: 模型权重路径
- `--top_k`: 检测的实体数量
- `--threshold`: 实体相似度阈值
- `--using_hard_prompt`: 是否使用硬提示
- `--soft_prompt_first`: 软提示位置

### 4.3 批量推理

```bash
python infer_by_batch.py \
    --prompt_ensemble \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

### 4.4 模型评估

#### COCO数据集评估
```bash
bash scripts/eval_coco.sh train_coco 0 '' 14
# 参数：实验名 GPU_ID 额外参数 权重epoch
```

#### Flickr30k数据集评估
```bash
bash scripts/eval_flickr30k.sh train_flickr30k 0 '' 29
```

#### NoCaps跨域评估
```bash
bash scripts/eval_nocaps.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
```

#### 语言评估（使用预生成结果）
```bash
bash scripts/language_eval.sh ./checkpoints/train_coco/overall_generated_captions.json
```

## 五、代码关键点解析

### 5.1 映射网络（MappingNetwork）

```python
# ClipCap.py:122-153
class MappingNetwork:
    def forward(self, x):
        # 1. CLIP特征投影
        x = self.linear(x).view(batch, clip_project_length, d_model)
        
        # 2. 可学习前缀
        prefix = self.prefix_const.expand(batch, prefix_length, d_model)
        
        # 3. Transformer处理
        inputs = torch.cat((x, prefix), dim=1)
        outputs = self.transformer(inputs)
        
        # 4. 返回软提示
        return outputs[:, clip_project_length:, :]
```

### 5.2 硬提示构建

```python
# utils.py:55-74
def compose_discrete_prompts(tokenizer, entities):
    # 格式：There are [entity1], [entity2], ... in image.
    prompt = "There are"
    for entity in entities:
        prompt += " " + entity + ","
    prompt = prompt[:-1] + " in image."
    return tokenizer.encode(prompt)
```

### 5.3 实体检测

```python
# retrieval_categories.py (推断)
# 1. 计算图像-文本相似度
logits = image_text_similarity(texts_embeddings, image_features)

# 2. Top-K检索
detected_objects = top_k_categories(entities_text, logits, top_k, threshold)
```

### 5.4 前向传播

```python
# ClipCap.py:204-241
def forward(self, continuous_prompt, caption_tokens, hard_prompts_length):
    # 1. 生成软提示
    continuous_embeddings = self.mapping_network(continuous_prompt)
    
    # 2. 词嵌入
    caption_embeddings = self.word_embed(caption_tokens)
    
    # 3. 组合提示
    if soft_prompt_first:
        embeddings = [continuous_embeddings, caption_embeddings]
    else:
        embeddings = [caption_embeddings, continuous_embeddings]
    
    # 4. GPT生成
    outputs = self.gpt(inputs_embeds=embeddings)
    return outputs
```

## 六、文件结构说明

```
ViECap/
├── main.py                    # 训练主脚本
├── ClipCap.py                 # 模型架构定义
├── CaptionsDataset.py         # 数据集类
├── utils.py                   # 工具函数
├── infer_by_instance.py       # 单图推理
├── infer_by_batch.py          # 批量推理
├── validation.py              # 评估脚本
├── entities_extraction.py     # 实体提取
├── texts_features_extraction.py  # 文本特征提取
├── images_features_extraction.py  # 图像特征提取
├── generating_prompt_ensemble.py # 提示集合生成
├── retrieval_categories.py    # 实体检索
├── search.py                  # 搜索策略（贪心/束搜索）
├── load_annotations.py        # 标注加载
├── scripts/                   # 训练/评估脚本
│   ├── train_coco.sh
│   ├── train_flickr30k.sh
│   ├── eval_coco.sh
│   ├── eval_flickr30k.sh
│   └── eval_nocaps.sh
├── annotations/               # 数据目录（需下载）
├── checkpoints/               # 模型权重（需下载）
└── requirements.txt           # 依赖列表
```

## 七、常见问题

### Q1: 训练时内存不足？
- 减小批次大小：`--bs 40`
- 使用预提取特征：`--using_clip_features`
- 启用混合精度：`--use_amp`

### Q2: 推理速度慢？
- 使用预提取图像特征
- 减小束搜索宽度：`--beam_width 3`
- 使用贪心搜索：`--using_greedy_search`

### Q3: 如何更换实体词汇表？
- 修改`--name_of_entities_text`参数
- 支持：`visual_genome_entities`, `coco_entities`, `open_image_entities`等

### Q4: 如何只使用软提示？
- 移除`--using_hard_prompt`参数
- 或使用`--only_hard_prompt`只使用硬提示

## 八、快速开始示例

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（从Releases）
# - checkpoints/
# - annotations/

# 3. 单图推理（使用预训练模型）
python infer_by_instance.py \
    --prompt_ensemble \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt

# 4. 训练（如果有数据）
bash scripts/train_coco.sh 0

# 5. 评估
bash scripts/eval_coco.sh train_coco 0 '' 14
```

## 九、模型性能

根据论文报告，ViECap在以下任务上的表现：

| 任务 | CIDEr | BLEU@4 | METEOR | SPICE |
|------|-------|--------|--------|-------|
| COCO → NoCaps (Overall) | 66.2 | - | - | 9.5 |
| COCO → Flickr30k | 38.4 | 17.4 | 18.0 | 11.2 |
| Flickr30k → COCO | 54.2 | 12.6 | 19.3 | 12.5 |
| COCO (In-domain) | 92.9 | 27.2 | 24.8 | 18.2 |

## 十、参考文献

```bibtex
@InProceedings{Fei_2023_ICCV,
    author    = {Fei, Junjie and Wang, Teng and Zhang, Jinrui and He, Zhenyu and Wang, Chengjie and Zheng, Feng},
    title     = {Transferable Decoding with Visual Entities for Zero-Shot Image Captioning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3136-3146}
}
```

---

**注意**：本项目主要在Linux环境下运行，Windows环境可能需要调整路径和脚本格式。











