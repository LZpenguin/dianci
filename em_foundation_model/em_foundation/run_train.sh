#!/usr/bin/env 
# run_train.sh: 多模态训练脚本示例代码，仅供参考。

set -e  # 脚本遇到错误立即退出

# 数据目录，包含 .h5/.json
DATA_DIR="/mnt/gold/lz/competitions/dianci/data/train_data"
# 本地 LLM 模型目录
LLM_DIR="/mnt/gold/lz/competitions/dianci/em_foundation_model/9G4B"
# EM 编码器权重
EM_ENCODER_WEIGHTS="/mnt/gold/lz/competitions/dianci/em_foundation_model/em_foundation/weight"

# 输出目录
OUTPUT_DIR="/mnt/gold/lz/competitions/dianci/models"
# 项目根目录
PROJECT_DIR="/mnt/gold/lz/competitions/dianci/em_foundation_model/em_foundation"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=4

# 用 nohup 启动并重定向日志，最后加 & 放到后台
torchrun --nproc_per_node=1 --master_port=12345 \
  train_mllm.py \
    --task-dir "$DATA_DIR" \
    --signal-encoder-path "$EM_ENCODER_WEIGHTS" \
    --llm-model-path   "$LLM_DIR" \
    --output-dir       "$OUTPUT_DIR" \
    --batch-size 8 \
    --max-length 2048 \
    --epochs 10 \
    --use-lora \
    --wandb-run-name "test"