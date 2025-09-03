#!/bin/bash
echo "开始上传数据..."
scp -P 2206 -r ./data/ zbtrs@115.190.124.192:/home/zbtrs/competitions/dianci/data/

echo "开始上传9G4B模型..."
scp -P 2206 -r ./em_foundation_model/9G4B/ zbtrs@115.190.124.192:/home/zbtrs/competitions/dianci/em_foundation_model/9G4B/

echo "开始上传em_foundation模型权重..."
scp -P 2206 -r ./em_foundation_model/em_foundation/weight/ zbtrs@115.190.124.192:/home/zbtrs/competitions/dianci/em_foundation_model/em_foundation/weight/
