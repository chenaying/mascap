# cocoeval.py 错误修复说明

## 错误信息

```
cocoeval.py: error: unrecognized arguments: checkpoints/train_coco/coco_generated_captions_meacap.json
```

## 问题原因

1. **通配符匹配问题**：`eval_coco.sh` 中使用 `coco*.json` 通配符，可能匹配到多个文件
   - `coco_generated_captions.json`（validation.py 生成的）
   - `coco_generated_captions_meacap.json`（validation_meacap.py 生成的）
   
2. **脚本调用错误**：
   - `eval_coco.sh` 调用的是 `validation.py`，应该生成 `coco_generated_captions.json`
   - 但如果目录中已有 `coco_generated_captions_meacap.json`，通配符会匹配到它

3. **参数格式问题**：`cocoeval.py` 可能不支持通配符展开后的多个文件路径

## 解决方案

### 方案 1：使用明确文件路径（推荐）

修改 `eval_coco.sh` 中的评估命令：

```bash
# 修改前
python evaluation/cocoeval.py --result_file_path $COCO_OUT_PATH/coco*.json

# 修改后（已修复）
RESULT_FILE="$COCO_OUT_PATH/coco_generated_captions.json"
python evaluation/cocoeval.py --result_file_path "$RESULT_FILE"
```

### 方案 2：手动指定文件路径

直接运行评估命令，明确指定文件：

```bash
# 使用 validation.py 生成的文件
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco_generated_captions.json

# 或使用 validation_meacap.py 生成的文件
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco_generated_captions_meacap.json
```

### 方案 3：清理旧文件

如果目录中有不需要的文件，先清理：

```bash
# 查看所有匹配的文件
ls -lh checkpoints/train_coco/coco*.json

# 删除不需要的文件（谨慎操作）
rm checkpoints/train_coco/coco_generated_captions_meacap.json

# 然后重新运行
bash scripts/eval_coco.sh train_coco 0 '' 14
```

## 使用正确的脚本

### 对于原始 ViECap（使用 validation.py）

```bash
bash scripts/eval_coco.sh train_coco 0 '' 14
# 生成：checkpoints/train_coco/coco_generated_captions.json
```

### 对于 MeaCap（使用 validation_meacap.py）

```bash
# 使用原始 Retrieve-then-Filter
bash scripts/eval_coco_meacap.sh train_coco 0 '' 14
# 生成：checkpoints/train_coco/coco_generated_captions_meacap.json

# 使用 EF 模块
bash scripts/eval_coco_meacap.sh train_coco 0 \
  '--use_entity_filtering --ef_filter_method threshold --ef_threshold 1' \
  14
```

## 验证文件存在

运行前检查文件是否存在：

```bash
# 检查 validation.py 生成的文件
ls -lh checkpoints/train_coco/coco_generated_captions.json

# 检查 validation_meacap.py 生成的文件
ls -lh checkpoints/train_coco/coco_generated_captions_meacap.json
```

## 快速修复命令

如果当前目录中有 `coco_generated_captions_meacap.json`，可以：

```bash
# 选项 1：直接评估 validation.py 生成的文件
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco_generated_captions.json

# 选项 2：评估 validation_meacap.py 生成的文件（使用 MeaCap 方法）
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco_generated_captions_meacap.json
```

## 注意事项

1. **文件命名区别**：
   - `validation.py` → `coco_generated_captions.json`（无后缀）
   - `validation_meacap.py` → `coco_generated_captions_meacap.json`（有 `_meacap` 后缀）

2. **脚本对应关系**：
   - `eval_coco.sh` → 调用 `validation.py`
   - `eval_coco_meacap.sh` → 调用 `validation_meacap.py`

3. **通配符问题**：
   - 如果目录中有多个匹配的文件，通配符会展开为多个参数
   - `cocoeval.py` 可能只接受单个文件路径

## 已修复的脚本

`scripts/eval_coco.sh` 已更新，现在会：
1. 首先查找 `coco_generated_captions.json`
2. 如果找不到，自动搜索替代文件
3. 使用明确的文件路径，避免通配符问题

