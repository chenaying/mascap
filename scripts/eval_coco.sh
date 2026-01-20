SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4
WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
COCO_OUT_PATH=checkpoints/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

COCO_LOG_FILE="$LOG_FOLDER/COCO_${TIME_START}.log"

python validation.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.4 \
--using_image_features \
--name_of_datasets coco \
--path_of_val_datasets ./annotations/coco/test_captions.json \
--name_of_entities_text coco_entities \
--image_folder ./annotations/coco/val2014/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$COCO_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
$OTHER_ARGS \
|& tee -a  ${COCO_LOG_FILE}

echo "==========================COCO EVAL================================"
# Use specific file name to avoid matching multiple files
RESULT_FILE="$COCO_OUT_PATH/coco_generated_captions.json"
if [ ! -f "$RESULT_FILE" ]; then
    echo "Warning: Result file not found: $RESULT_FILE"
    echo "Searching for alternative files..."
    RESULT_FILE=$(find $COCO_OUT_PATH -name "coco_generated_captions.json" | head -1)
    if [ -z "$RESULT_FILE" ]; then
        echo "Error: No result file found in $COCO_OUT_PATH"
        exit 1
    fi
fi
echo "Evaluating: $RESULT_FILE"
python evaluation/cocoeval.py --result_file_path "$RESULT_FILE" |& tee -a  ${COCO_LOG_FILE}