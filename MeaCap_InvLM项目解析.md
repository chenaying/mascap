# MeaCap_InvLM 项目解析

## 1. 项目定位

`MeaCap_InvLM` 可以理解为：以 ViECap 为主体框架，叠加 MeaCap 的 Retrieve-then-Filter / EF 思路进行改造，用于图像描述（Image Captioning）任务。

核心目标是：输入图像，输出自然语言描述句子。

---

## 2. 核心方法与技术路线

### 2.1 基础生成链路

项目的主干生成链路是：

1. 使用 CLIP 提取图像（或文本）特征；
2. 通过 Mapping Network 将特征映射到语言模型可用的连续提示（soft prompt）空间；
3. 与可选的离散实体提示（hard prompt）拼接；
4. 送入 GPT/OPT 进行自回归解码生成 caption。

### 2.2 本分支改造方向

相较原版 ViECap，本项目在评估/推理流程中引入了 MeaCap 风格模块：

- 新增或改造了 `validation_meacap.py`、`viecap_inference_adapted.py`；
- 将实体获取方式从“纯类别词表匹配”扩展为“memory caption 检索 + 过滤”；
- 支持结合检索记忆库、文本图解析器与句向量模型进行概念提取。

---

## 3. 代码结构总览（按功能）

- 训练入口：`main.py`
- 原版评估入口：`validation.py`
- MeaCap 改造评估入口：`validation_meacap.py`
- 单图推理（改造版）：`viecap_inference_adapted.py`
- 模型定义：`ClipCap.py`
- 训练数据集封装：`CaptionsDataset.py`
- 搜索解码策略：`search.py`
- 实体过滤工具：`utils/entity_filtering_utils.py`
- 脚本目录：`scripts/`（训练、评估、语言评测）

---

## 4. 训练流程解析

训练由 `main.py` 驱动，关键步骤如下：

1. 读取参数（batch size、epoch、prompt 长度、模型类型等）；
2. 构建 `CaptionsDataset`，读取带实体信息的数据集；
3. 初始化 `ClipCaptionModel` 或 `ClipCaptionPrefix`；
4. 前向计算 + 交叉熵损失；
5. 定期保存 `latest` 与 `epoch` 权重到 `checkpoints/`。

训练脚本示例（COCO）：

- `scripts/train_coco.sh`

---

## 5. 评估与推理流程解析

### 5.1 原版评估（`validation.py`）

原版流程主要是：

1. 读取图像特征或原图；
2. 基于实体词表 embedding 与图像特征做相似度匹配；
3. 选 top-k 实体构造 hard prompt；
4. 调用 `beam_search` 或 `greedy_search` 生成描述；
5. 输出 json 结果，再配合 `evaluation/cocoeval.py` 计算指标。

### 5.2 MeaCap 改造评估（`validation_meacap.py`）

改造流程主要是：

1. 从 memory bank 中加载 caption 与 embedding；
2. 对输入图像检索 top-k memory captions；
3. 对检索到的 captions 执行过滤/概念提取；
4. 组合 hard prompt + soft prompt；
5. 进行文本解码并保存 `_meacap` 后缀结果文件。

对应脚本示例：

- `scripts/eval_coco_meacap.sh`

### 5.3 单图推理（`viecap_inference_adapted.py`）

支持单张图输入，具备以下能力：

- 优先读取本地模型路径（兼容离线）；
- 加载检索用 CLIP、SentenceBERT、Flan-T5 parser；
- 从 memory bank 检索并提取概念；
- 生成最终 caption。

---

## 6. 关键模块说明

### 6.1 `ClipCap.py`

- 定义 MappingNetwork（将 CLIP 特征投影到 LM 隐空间）；
- 定义 `ClipCaptionModel`（支持 hard/soft prompt 组合）；
- 支持 GPT 与 OPT 语言模型后端。

### 6.2 `CaptionsDataset.py`

- 读取带实体训练样本；
- 支持直接使用预提取特征（`--using_clip_features`）；
- 负责 token 对齐、mask 构建、hard prompt 拼接准备。

### 6.3 `search.py`

包含多种解码策略：

- `greedy_search`
- `beam_search`
- `opt_search`
- `contrastive_search`
- `magic_search`

---

## 7. 目前仓库状态中的风险点（建议优先处理）

1. `README.md` 顶部存在冲突标记（`<<<<<<< HEAD`），说明存在未清理的 merge 冲突痕迹；
2. 代码引用了 `utils.detect_utils`、`models.clip_utils` 等模块，但当前目录下未见对应实现文件，部分流程可能运行时报 `ImportError`；
3. `utils.py` 与 `utils/` 包并存，当前通过 `utils/__init__.py` 做了兼容导入，能缓解冲突但维护复杂度偏高。

---

## 8. 结论

该项目是“ViECap 主体 + MeaCap 检索过滤改造”的图像描述实验工程。整体链路覆盖训练、评估、推理，但在可复现性上仍依赖：

- 完整的外部依赖模型/权重；
- 补齐缺失模块；
- 清理冲突遗留内容。

在进入新一轮实验前，建议先完成一次“最小可运行验证”（单脚本、单 checkpoint、单数据子集）。
