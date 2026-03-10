# `eval_flickr30k_meacap.sh` 脚本解析

## 1. 脚本定位

`scripts/eval_flickr30k_meacap.sh` 是 Flickr30k 数据集的评估入口脚本，用于调用 `validation_meacap.py` 执行基于 MeaCap（Retrieve-then-Filter / EF 思路）的 caption 生成，并进一步调用 `evaluation/cocoeval.py` 计算指标。

该脚本作用是把“生成结果 + 指标评测 + 日志记录”串成一次完整流程。

---

## 2. 参数接口说明

脚本接收 4 个位置参数：

- `$1 -> EXP_NAME`：实验名（通常对应 `checkpoints/<EXP_NAME>` 目录）
- `$2 -> DEVICE`：GPU 编号（如 `0`）
- `$3 -> OTHER_ARGS`：额外参数字符串（透传给 `validation_meacap.py`）
- `$4 -> EPOCH`：权重编号（如 `14` 对应 `coco_prefix-0014.pt`）

脚本中对应变量定义：

- `WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt`
- `FLICKR_OUT_PATH=checkpoints/$EXP_NAME`

---

## 3. 执行流程分解

### 步骤 1：切换到项目根目录

通过 `SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)` 定位脚本目录，再 `cd $SHELL_FOLDER/..` 回到仓库根目录，确保相对路径有效。

### 步骤 2：准备日志目录与日志文件

- 日志目录：`logs/${EXP_NAME}_EVAL_MEACAP_FLICKR`
- 日志文件：`FLICKR_MEACAP_<时间戳>.log`

使用 `mkdir -p` 自动创建目录，避免首次运行失败。

### 步骤 3：调用 `validation_meacap.py` 生成预测结果

关键固定参数：

- 数据集：`--name_of_datasets flickr30k`
- 标注：`--path_of_val_datasets ./annotations/flickr30k/test_captions.json`
- 图像目录：`--image_folder ./annotations/flickr30k/flickr30k-images/`
- 使用预提取图像特征：`--using_image_features`
- 使用 hard prompt：`--using_hard_prompt --soft_prompt_first`
- 检索与过滤相关模型：
  - `--vl_model openai/clip-vit-base-patch32`
  - `--parser_checkpoint lizhuang144/flan-t5-base-VG-factual-sg`
  - `--wte_model_path sentence-transformers/all-MiniLM-L6-v2`
- memory bank：
  - `--memory_id coco`
  - `--memory_caption_path data/memory/coco/memory_captions.json`
  - `--memory_caption_num 5`

同时透传 `$OTHER_ARGS`，便于外部覆盖默认设置（例如启用 EF 的额外开关）。

### 步骤 4：调用 `cocoeval.py` 计算指标

对 `checkpoints/$EXP_NAME/` 下匹配 `flickr30k*_meacap.json` 的结果文件进行评测，输出指标并追加到同一日志文件。

---

## 4. 输入与输出

### 输入依赖

- 模型权重：`checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt`
- Flickr30k 标注：`./annotations/flickr30k/test_captions.json`
- Flickr30k 图像目录：`./annotations/flickr30k/flickr30k-images/`
- memory bank 及其 embedding（由 `validation_meacap.py` 间接读取）

### 输出文件

- 生成结果：`checkpoints/$EXP_NAME/flickr30k*_meacap.json`
- 评估日志：`logs/${EXP_NAME}_EVAL_MEACAP_FLICKR/FLICKR_MEACAP_<时间戳>.log`

---

## 5. 典型用法

示例：

```bash
bash scripts/eval_flickr30k_meacap.sh train_coco 0 "--using_greedy_search" 14
```

含义：

- 使用 `train_coco` 实验目录下第 `14` 轮权重；
- 在 `cuda:0` 上评估；
- 额外启用 `--using_greedy_search`；
- 结果和日志落到对应 `checkpoints` 与 `logs` 目录。

---

## 6. 注意事项与潜在问题

1. 脚本默认 `--memory_id coco`，即使用 COCO memory bank 做 Flickr30k 评估，这是跨域设定，需与实验目标保持一致。
2. `--using_image_features` 要求预提取特征文件存在，否则会在验证脚本中找不到输入。
3. `WEIGHT_PATH` 命名格式固定为 `coco_prefix-00${EPOCH}.pt`，若权重命名不同需手动调整。
4. `$OTHER_ARGS` 建议用双引号包裹并整体传入，避免参数被 shell 错误拆分。
5. 日志采用 `tee -a` 追加模式，多次运行同名日志文件会持续追加内容（当前脚本用时间戳规避了文件名冲突）。

---

## 7. 一句话总结

该脚本是 Flickr30k 的 MeaCap 评估一键入口：负责目录切换、参数组装、结果生成、指标评测和日志归档，适合用于标准化复现实验流程。
