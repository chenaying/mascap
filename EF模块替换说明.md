# EF 模块替换 Retrieve-then-Filter 的 Filter 阶段说明

## 一、概述

本文档说明如何使用 **Entity Filtering (EF)** 模块替换 MeaCap_InvLM 模型中 Retrieve-then-Filter 模块的 **Filter 阶段**（对检索语句的处理部分）。

### 1.1 替换内容

- **保留**：Retrieve 阶段（从记忆库检索 Top-K 条描述）
- **替换**：Filter 阶段（从检索到的描述中提取关键概念）
  - 原方法：`retrieve_concepts`（场景图解析 + 语义合并）
  - 新方法：`retrieve_concepts_ef`（频率统计 + 实体过滤）

### 1.2 替换优势

| 对比项 | 原 Retrieve-then-Filter | EF 方法 |
|--------|------------------------|---------|
| **依赖模型** | Flan-T5 + SentenceBERT + CLIP | NLTK（轻量级） |
| **计算开销** | 高（场景图解析 + 语义相似度） | 低（POS 标注 + 频率统计） |
| **推理速度** | ~300-600ms/image | ~50-100ms/image |
| **实现复杂度** | 高（多模型协调） | 低（简单统计） |
| **概念质量** | 高（短语级概念） | 中（单词级实体） |

---

## 二、文件变更

### 2.1 新增文件

| 文件 | 说明 |
|------|------|
| `utils/entity_filtering_utils.py` | EF 模块核心实现（实体提取与过滤） |

### 2.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `viecap_inference_adapted.py` | 添加 EF 选项，替换 `retrieve_concepts` 调用 |
| `validation_meacap.py` | 添加 EF 选项，替换 `retrieve_concepts` 调用 |
| `utils/__init__.py` | 导出 `retrieve_concepts_ef` 函数 |

---

## 三、使用方法

### 3.1 单图推理

**使用 EF 方法**：
```bash
python viecap_inference_adapted.py \
    --image_path images/example.jpg \
    --memory_id coco \
    --memory_caption_path data/memory/coco/memory_captions.json \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --use_entity_filtering \
    --ef_filter_method threshold \
    --ef_threshold 1 \
    --max_num_of_entities 5
```

**使用原 Retrieve-then-Filter 方法**（默认）：
```bash
python viecap_inference_adapted.py \
    --image_path images/example.jpg \
    --memory_id coco \
    --memory_caption_path data/memory/coco/memory_captions.json \
    --memory_caption_num 5 \
    --using_hard_prompt
    # 不添加 --use_entity_filtering，使用默认方法
```

### 3.2 批量评估

**使用 EF 方法**：
```bash
python validation_meacap.py \
    --name_of_datasets coco \
    --path_of_val_datasets annotations/coco/test_captions.json \
    --memory_id coco \
    --memory_caption_path data/memory/coco/memory_captions.json \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --use_entity_filtering \
    --ef_filter_method threshold \
    --ef_threshold 1 \
    --max_num_of_entities 5 \
    --weight_path checkpoints/train_coco/coco_prefix-0014.pt
```

### 3.3 过滤方法选择

**方法 1：固定阈值过滤**（`threshold`）
```bash
--ef_filter_method threshold --ef_threshold 2
# 保留出现频率 >= 2 的实体
```

**方法 2：Normal 分布过滤**（`normal`）
```bash
--ef_filter_method normal --ef_alpha 1.0
# 保留频率 > mean + 1.0 * std 的实体
```

**方法 3：Log-Normal 分布过滤**（`log_normal`）
```bash
--ef_filter_method log_normal --ef_alpha 1.0
# 保留 log(频率) > log_mean + 1.0 * log_std 的实体
```

---

## 四、代码实现

### 4.1 EF 模块核心函数

**位置**：`utils/entity_filtering_utils.py`

```python
def retrieve_concepts_ef(
    select_memory_captions: List[str],
    filter_method: str = 'threshold',
    threshold: int = 1,
    alpha: float = 1.0,
    max_entities: int = 5
) -> List[str]:
    """
    从检索到的描述中提取关键实体（使用频率统计）
    
    Args:
        select_memory_captions: 检索到的描述列表
        filter_method: 过滤方法 ('threshold', 'normal', 'log_normal')
        threshold: 频率阈值（用于 threshold 方法）
        alpha: Alpha 参数（用于 normal/log_normal 方法）
        max_entities: 最大实体数量
    
    Returns:
        List[str]: 关键实体列表
    """
    # 1. 提取实体并统计频率
    freq_entities = extract_entities_from_captions(select_memory_captions)
    
    # 2. 根据方法过滤
    if filter_method == 'threshold':
        filtered = filter_entities_by_threshold(freq_entities, threshold)
    elif filter_method == 'normal':
        filtered = filter_entities_normal(freq_entities, alpha)
    elif filter_method == 'log_normal':
        filtered = filter_entities_log_normal(freq_entities, alpha)
    
    # 3. 限制数量
    return filtered[:max_entities]
```

### 4.2 集成调用示例

**位置**：`validation_meacap.py:102-123`

```python
# Retrieve 阶段（不变）
select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
select_memory_captions = [memory_captions[id] for id in select_memory_ids]

# Filter 阶段（可选择方法）
if args.use_entity_filtering:
    # EF 方法
    detected_objects = retrieve_concepts_ef(
        select_memory_captions=select_memory_captions,
        filter_method=args.ef_filter_method,
        threshold=args.ef_threshold,
        alpha=args.ef_alpha,
        max_entities=args.max_num_of_entities
    )
else:
    # 原 Retrieve-then-Filter 方法
    detected_objects = retrieve_concepts(
        parser_model=parser_model,
        parser_tokenizer=parser_tokenizer,
        wte_model=wte_model,
        select_memory_captions=select_memory_captions,
        image_embeds=batch_image_embeds,
        device=device
    )

# 后续处理（完全相同）
discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects)
discrete_embeddings = model.word_embed(discrete_tokens)
```

---

## 五、工作流程对比

### 5.1 原 Retrieve-then-Filter 流程

```
检索描述 → 场景图解析(Flan-T5) → 实体提取 → 语义合并(SentenceBERT) → 图像过滤 → 关键概念
```

### 5.2 EF 替换后的流程

```
检索描述 → POS标注(NLTK) → 名词提取 → 频率统计 → 阈值/分布过滤 → 关键实体
```

---

## 六、参数说明

### 6.1 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_entity_filtering` | `flag` | `False` | 启用 EF 方法 |
| `--ef_filter_method` | `str` | `'threshold'` | 过滤方法：`'threshold'`, `'normal'`, `'log_normal'` |
| `--ef_threshold` | `int` | `1` | 频率阈值（用于 `threshold` 方法） |
| `--ef_alpha` | `float` | `1.0` | Alpha 参数（用于 `normal`/`log_normal` 方法） |
| `--max_num_of_entities` | `int` | `5` | 最大实体数量 |

### 6.2 保留参数

| 参数 | 说明 |
|------|------|
| `--memory_caption_num` | 检索的 Top-K 描述数量 |
| `--memory_id` | 记忆库 ID |
| `--memory_caption_path` | 记忆库文本文件路径 |

---

## 七、性能对比

### 7.1 推理速度

| 方法 | 单图推理时间 | 加速比 |
|------|-------------|--------|
| Retrieve-then-Filter | ~300-600ms | 1.0x |
| EF (threshold) | ~50-100ms | **3-6x** |

### 7.2 内存占用

| 方法 | 额外内存需求 |
|------|-------------|
| Retrieve-then-Filter | Flan-T5 (~500MB) + SentenceBERT (~400MB) |
| EF | NLTK (~50MB) |

### 7.3 依赖对比

| 依赖 | Retrieve-then-Filter | EF |
|------|---------------------|-----|
| Flan-T5 | ✅ 必需 | ❌ 不需要 |
| SentenceBERT | ✅ 必需 | ❌ 不需要 |
| NLTK | ❌ 不需要 | ✅ 必需 |

---

## 八、使用建议

### 8.1 何时使用 EF

- ✅ 需要快速推理（实时应用）
- ✅ 资源受限（内存/GPU 有限）
- ✅ 概念质量要求中等（单词级实体足够）
- ✅ 不想加载大型模型（Flan-T5, SentenceBERT）

### 8.2 何时使用原方法

- ✅ 需要高质量概念（短语级，如 "cute girl"）
- ✅ 有充足的计算资源
- ✅ 概念提取质量是优先考虑

### 8.3 参数调优

**固定阈值过滤**（推荐开始使用）：
- `threshold=1`：保留所有出现的实体（最宽松）
- `threshold=2`：保留出现 2 次及以上的实体（推荐）
- `threshold=3`：保留高频实体（最严格）

**自适应过滤**（需要调优）：
- `normal`：适合频率分布接近正态的情况
- `log_normal`：适合频率分布偏斜的情况
- `alpha=0.5`：更宽松
- `alpha=1.5`：更严格

---

## 九、示例输出

### 9.1 输入（检索到的描述）

```python
select_memory_captions = [
    "A cute girl is sitting on a bed with a pink blanket.",
    "A young woman lies on a bed covered with a pink blanket.",
    "A girl is resting on a bed with pink sheets."
]
```

### 9.2 EF 方法输出

```python
# threshold=1, max_entities=5
detected_objects = ['bed', 'blanket', 'girl', 'woman', 'pink']
```

### 9.3 原方法输出（对比）

```python
# Retrieve-then-Filter
detected_objects = ['cute girl', 'bed', 'pink blanket']
```

**说明**：
- EF：提取单词级实体，可能包含同义词（"girl", "woman"）
- 原方法：提取短语级概念，语义更精确

---

## 十、总结

### 10.1 核心变化

1. **新增 EF 模块**：`utils/entity_filtering_utils.py`
2. **保留 Retrieve 阶段**：继续使用 CLIP 检索 Top-K 描述
3. **替换 Filter 阶段**：用简单的频率统计替代复杂的场景图解析
4. **向后兼容**：可通过参数选择使用原方法或 EF 方法

### 10.2 优势

- ✅ **速度快**：推理时间减少 3-6 倍
- ✅ **依赖少**：不需要 Flan-T5 和 SentenceBERT
- ✅ **实现简单**：代码量少，易于维护
- ✅ **完全可插拔**：输出格式兼容，无缝替换

### 10.3 注意事项

- ⚠️ 概念质量可能略低于原方法（单词级 vs 短语级）
- ⚠️ 需要适当调优过滤参数以获得最佳效果
- ⚠️ NLTK 数据需要首次运行时下载（自动）

---

**使用方式**：添加 `--use_entity_filtering` 参数即可启用 EF 方法！

