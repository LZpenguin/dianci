#!/usr/bin/env 
# run_train.sh: 多模态训练脚本示例代码，仅供参考。

export HF_HOME='/dev/shm/hf_cache'

set -e  # 脚本遇到错误立即退出

BASE_DIR="/home/zbtrs/competitions/dianci"

# 数据目录，包含 .h5/.json
DATA_DIR="${BASE_DIR}/data/train_data"
# 本地 LLM 模型目录
LLM_DIR="${BASE_DIR}/em_foundation_model/9G4B"
# EM 编码器权重
EM_ENCODER_WEIGHTS="${BASE_DIR}/em_foundation_model/em_foundation/weight"
# 继续训练的模型路径
TRAINED_DIR="/home/zbtrs/competitions/dianci/models/v32_b64_oc_p4_e3/checkpoint-49500"


# 输出目录
OUTPUT_DIR="${BASE_DIR}/models"
# 项目根目录
PROJECT_DIR="${BASE_DIR}/em_foundation_model/em_foundation"
# wandb 运行名称
WANDB_RUN_NAME="v33_b64_oc_p4_e3+1fe"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 用 nohup 启动并重定向日志，最后加 & 放到后台
torchrun --nproc_per_node=4 --master_port=12345 \
  train_mllm.py \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --task-dir "$DATA_DIR" \
    --signal-encoder-path "$EM_ENCODER_WEIGHTS" \
    --llm-model-path   "$LLM_DIR" \
    --trained-dir "$TRAINED_DIR" \
    --output-dir       "$OUTPUT_DIR/$WANDB_RUN_NAME" \
    --batch-size 12 \
    --max-length 256 \
    --epochs 3 \
    --lr 1e-4 \
    --use-lora \
    --freeze-signal-encoder