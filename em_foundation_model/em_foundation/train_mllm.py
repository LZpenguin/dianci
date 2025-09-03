#!/usr/bin/env python3
"""
train_multimodal.py: 多模态训练脚本示例代码，仅供参考。
"""

import os
import json
import h5py
import torch
import glob
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from argparse import ArgumentParser
import sys
import logging
import random
from torch.utils.data import SequentialSampler
from safetensors.torch import load_file
from accelerate import DistributedDataParallelKwargs
import wandb
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EM_BASE = os.path.join(
    SCRIPT_DIR,
    "em_foundation"
)
sys.path.insert(0, EM_BASE)
from models.configuration_em import SiTMAEConfig
from models.modeling_em   import SiTMAEModelWithoutMask as SiTMAEModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset

LOG_FILENAME = os.path.join(".", 'train.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),             
        logging.FileHandler(LOG_FILENAME, mode='a')  
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--llm-model-path', type=str, required=True,
                        help='本地 LLM 模型和 tokenizer 目录')
    parser.add_argument('--task-dir', type=str, default=None,
                        help='存放所有 h5/json 文件的目录，按同名配对')
    parser.add_argument('--task-files', nargs='+',
                        help='成对传入 h5 和 json 路径，如: data_modulation.h5 data_modulation.json')
    parser.add_argument('--signal-encoder-path', type=str, required=True,
                        help='信号编码器权重路径，例如 signal_encoder.pth')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=512,
                        help='文本部分最大 token 数，所有 prompt 文本将被截断到此长度 (<= 模型最大 context)')
    parser.add_argument('--freeze-llm', action='store_true',
                        help='第一阶段冻结 LLM，仅训练投影层和/或信号编码器')
    parser.add_argument('--freeze-signal-encoder', action='store_true',
                        help='第一阶段冻结信号编码器，仅训练投影层')
    parser.add_argument('--use-lora', action='store_true',
                        help='是否在大语言模型上使用 LoRA（第二阶段微调时开启）')
    parser.add_argument('--wandb-project', type=str, default='em-foundation-model',
                        help='Wandb项目名称')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb运行名称')

    return parser.parse_args()


class MultiModalDataset(Dataset):
    def __init__(self, task_files, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_files = task_files
        # —— 预加载：把每个 .h5 中的 IQ_data 一次性读入内存
        self.h5_data = {}
        for h5_path, json_path in task_files:
            with h5py.File(h5_path, 'r') as f:
                arr = f['IQ_data'][:]                 
            self.h5_data[h5_path] = torch.from_numpy(arr) 
            entries = json.load(open(json_path, 'r', encoding='utf-8'))
            for item in entries:
                item['h5_path'] = h5_path
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        h5_path = item['h5_path']
        idx_in_h5 = item['iq_index_in_h5']
        # 直接切片取已经加载到内存的 Tensor
        iq = self.h5_data[h5_path][idx_in_h5]     
        m = iq.abs().max().clamp(min=1e-6)
        iq = iq / m                                  
        # 文本编码：仅 question + answer（后面模型 forward 里再插 IQ special tokens）
        question = item['question']
        answer   = item['answer'] if 'answer' in item else ''
        prompt = question + self.tokenizer.eos_token + ((answer + self.tokenizer.eos_token) if 'answer' in item else '')
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return {'iq': iq, 'input_ids': input_ids, 'attention_mask': attention_mask}


class DynamicIQCollator:
    """
    按 batch 内最长 IQ 自动 pad/truncate，
    并对文本做常规 padding。
    """
    def __init__(self, tokenizer, patch_size):
        self.tokenizer = tokenizer
        self.patch_size = patch_size

    def __call__(self, batch):
        # 文本部分 padding
        input_ids = [b['input_ids'] for b in batch]
        attention_mask = [b['attention_mask'] for b in batch]
        input_ids = pad_sequence(input_ids, batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True,
                                      padding_value=0)

        # IQ 部分按最大长度 pad/truncate
        iq_list = [b['iq'] for b in batch] 
        max_len = max(iq.shape[0] for iq in iq_list)
        # 确保可整除
        rem = max_len % self.patch_size[0]
        if rem:
            max_len += (self.patch_size[0] - rem)
        padded = []
        for iq in iq_list:
            L = iq.shape[0]
            if L >= max_len:
                padded.append(iq[:max_len])
            else:
                pad = torch.zeros(max_len - L, iq.shape[1],
                                  dtype=iq.dtype, device=iq.device)
                padded.append(torch.cat([iq, pad], dim=0))
        iq_batch = torch.stack(padded, dim=0)

        return {
            'iq': iq_batch,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def load_sit_encoder_model(config: SiTMAEConfig, model_dir: str):
    """
    按 key 对齐加载 SiTMAEModel 权重，并裁剪 position_embeddings 到模型定义长度（N token）。
    model_dir 下需包含 model.safetensors。
    """
    # 1. 实例化纯骨干
    model = SiTMAEModel(config)
    # 2. 读取 safetensors 权重
    pretrained = load_file(os.path.join(model_dir, "model.safetensors"))
    # 3. 拿当前 model 的 state_dict
    model_dict = model.state_dict()
    # 4. 对齐并裁剪 position_embeddings
    for key in model_dict.keys():
        if key not in pretrained:
            continue
        if key == "sit.embeddings.position_embeddings":
            # 模型定义的 position_embeddings 长度
            tgt = model_dict[key].shape[0]
            # 预训练中如果更长就裁剪前 tgt 条
            pretrained[key] = pretrained[key][:tgt]
        model_dict[key] = pretrained[key]
    # 5. 加载对齐后的权重
    model.load_state_dict(model_dict)
    return model


class MultiModalModel(nn.Module):
    def __init__(self, signal_encoder, llm, tokenizer, use_lora=False,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.signal_encoder = signal_encoder
        # 特殊 token id，供 forward 时插入
        self.iq_start_id = tokenizer.convert_tokens_to_ids("<IQ_START>")
        self.iq_end_id   = tokenizer.convert_tokens_to_ids("<IQ_END>")
        self.embed_layer = llm.get_input_embeddings()
        # ===== 两层 MLP：EM-hidden → mlp_dim → LLM-hidden =====
        em_hidden  = signal_encoder.config.hidden_size
        llm_hidden = llm.config.hidden_size
        mlp_dim    = llm_hidden * 4
        self.signal_to_hidden = nn.Sequential(
            nn.Linear(em_hidden, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, llm_hidden),
            nn.LayerNorm(llm_hidden),
        ).to(llm.device)
        # 根据 use_lora 决定是否注入 LoRA
        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['q_proj', 'v_proj'],
                lora_dropout=lora_dropout,
                bias='none',
                task_type='CAUSAL_LM'
            )
            self.llm = get_peft_model(llm, peft_config)
        else:
            self.llm = llm

    def forward(self, iq, input_ids, attention_mask):
        # 信号编码
        # —— 1) Patchify —— #
        B, L, C = iq.shape
        P = self.signal_encoder.config.patch_size[0]
        assert L % P == 0, f"IQ 长度 {L} 必须是 patch_size {P} 的整数倍"
        num_patches = L // P
        patches = iq.view(B, num_patches, P * C) 

        # —— 2) 构造位置和ID —— #
        patch_positions = torch.arange(num_patches, device=iq.device) \
                              .unsqueeze(0).expand(B, -1)     
        sample_ids_seq = patch_positions.clone().unbind(0)  

        # —— 3) 无 Mask 特征提取 —— #
        # 调用 WithoutMask.forward（return_dict=True），并解包出全量 patch 特征
        sit_out = self.signal_encoder(
            patches,
            patch_positions,
            list(sample_ids_seq),
            return_dict=True   
        )
        feats = sit_out.last_hidden_state       

        # —— 1) 投影成 signal prefix embeddings —— #
        prompt_embeds = self.signal_to_hidden(feats)  
        prompt_embeds = prompt_embeds.to(self.llm.dtype)
        # —— 强制转成 LLM 用的半精度 —— #
        prompt_embeds = prompt_embeds.to(self.llm.dtype)
        # —— 2) special-token embedding & expand —— #
        B, npatch, H = prompt_embeds.size()
        start_embed = self.embed_layer(
            torch.tensor([self.iq_start_id], device=iq.device)
        ).unsqueeze(0).expand(B, 1, H)
        end_embed   = self.embed_layer(
            torch.tensor([self.iq_end_id],   device=iq.device)
        ).unsqueeze(0).expand(B, 1, H)
        # —— 3) 文本 embedding —— #
        text_embeds = self.embed_layer(input_ids) 
        # —— 4) 拼 [IQ_START], signal, [IQ_END], question+answer —— #
        inputs_embeds = torch.cat([start_embed, prompt_embeds, end_embed, text_embeds], dim=1)
        # —— 5) attention mask 对齐 —— #
        prefix_len = 1 + npatch + 1
        prefix_mask = torch.ones(B, prefix_len,
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # —— 6) labels 拼接（prefix 全部忽略） —— #
        labels = torch.cat([
            torch.full((B, prefix_len), -100, device=input_ids.device),
            input_ids
        ], dim=1)
        # 前向
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # 初始化 Wandb（仅在主进程）
    if accelerator.is_local_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "max_length": args.max_length,
                "freeze_llm": args.freeze_llm,
                "freeze_signal_encoder": args.freeze_signal_encoder,
                "use_lora": args.use_lora,
                "llm_model_path": args.llm_model_path,
                "signal_encoder_path": args.signal_encoder_path,
            }
        )

    # TensorBoard 日志目录
    tb_log_dir = os.path.join(args.output_dir, "runs")
    writer = SummaryWriter(log_dir=tb_log_dir)

    # —— 加载 LLM —— 
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<IQ_START>", "<IQ_END>"]
    })
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    llm.resize_token_embeddings(len(tokenizer))
    # —— 实例化 EM 基础模型作为 signal_encoder —— 
    # 这里 SiTMAEConfig() 可根据需要传入自定义参数
    # —— 用专用函数加载并转换到“无 Mask”模型 —— 
    # 在创建 config 时打开 use_fs 并传入 fs 参数
    em_config = SiTMAEConfig(
        max_seq_len=(4096,1),
        use_fs=True,
        fs=20e6,      # e.g. 1e6 表示 1MHz
    )
    # 先加载带 Mask 的骨干并对齐权重
    base_encoder = load_sit_encoder_model(em_config, args.signal_encoder_path)
    # 切换到无 Mask 版本并拷贝权重
    signal_encoder = SiTMAEModel(base_encoder.config)
    signal_encoder.load_state_dict(base_encoder.state_dict(), strict=False)
    signal_encoder.to(device)

    # ==== 自动扫描 task-dir 下的文件，按同名 .h5/.json 配对 ====
    if args.task_dir:
        h5_paths = sorted(glob.glob(os.path.join(args.task_dir, '*.h5')))
        task_pairs = []
        for h5p in h5_paths:
            base = os.path.splitext(os.path.basename(h5p))[0]
            jsonp = os.path.join(args.task_dir, base + '.json')
            if not os.path.exists(jsonp):
                raise FileNotFoundError(f"找不到对应的 JSON 文件: {jsonp}")
            task_pairs.append((h5p, jsonp))
    else:
        task_pairs = [
            tuple(args.task_files[i:i+2])
            for i in range(0, len(args.task_files), 2)
        ]

    # 模型构建，可选 LoRA
    model = MultiModalModel(signal_encoder, llm, tokenizer, use_lora=args.use_lora)
    # 冻结 Base Model（保留 Adapter 可训练）
    if args.freeze_llm:
        # 如果使用了 LoRA，model.llm.base_model 是原始模型
        base = getattr(model.llm, 'base_model', model.llm)
        for name, param in base.named_parameters():
            param.requires_grad = False
    if args.freeze_signal_encoder:
        for param in model.signal_encoder.parameters():
            param.requires_grad = False
    # 加载 signal_encoder，并拿到 patch_size
    patch_size = signal_encoder.config.patch_size
    # 构建 Dataset
    full_dataset = MultiModalDataset(task_pairs, tokenizer, max_length=args.max_length)
    # 随机抽 100 条做验证，剩下训练
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    val_indices   = indices[:50]
    train_indices = indices[50:]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    # 实例化collator
    collator = DynamicIQCollator(tokenizer, patch_size)
    # DataLoader 里直接传入
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=32,              
        pin_memory=True,         
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=32,
        pin_memory=True,
        collate_fn=collator
    )
    # 优化器：仅对需要梯度的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    num_update_steps = len(train_loader) * args.epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_update_steps),
        num_training_steps=num_update_steps
    )
    # 只给训练用的 loader 做分布式封装，验证 loader 保留原始 sampler
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    model.train()
    global_step = 0
    
    # 创建总的训练进度条，仅在主进程显示
    total_steps = len(train_loader) * args.epochs
    if accelerator.is_local_main_process:
        overall_progress = tqdm(
            total=total_steps,
            desc="Training Progress",
            leave=True,
            dynamic_ncols=True
        )
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            outputs = model(batch['iq'], batch['input_ids'], batch['attention_mask'])
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # TensorBoard: 写入当前 loss
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            # Wandb: 记录训练指标
            if accelerator.is_local_main_process:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step
                }, step=global_step)
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新总进度条显示信息
            if accelerator.is_local_main_process:
                avg_loss = epoch_loss / (step + 1)
                current_lr = lr_scheduler.get_last_lr()[0]
                overall_progress.set_postfix({
                    'epoch': f'{epoch+1}/{args.epochs}',
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                overall_progress.update(1)

            if step % 50 == 0:
                avg = epoch_loss / (step + 1)
                if accelerator.is_local_main_process:
                    logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, Avg: {avg:.4f}")

            if global_step % 1000 == 0:
                # 1) 保存当前模型（unwrap 后才能 save_pretrained）
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                raw = accelerator.unwrap_model(model)
                raw.llm.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(raw.signal_to_hidden.state_dict(), os.path.join(ckpt_dir, "signal_to_hidden.pt"))
                torch.save(raw.signal_encoder.state_dict(),    os.path.join(ckpt_dir, "signal_encoder.pt"))
                
                # Wandb: 保存模型检查点作为artifact
                if accelerator.is_local_main_process:
                    artifact = wandb.Artifact(f"model-checkpoint-{global_step}", type="model")
                    artifact.add_dir(ckpt_dir)
                    wandb.log_artifact(artifact)

                # 2) 验证 & 打印，仅在主进程
                if accelerator.is_local_main_process:
                    logger.info(f"\n>>> Validation at step {global_step}")
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        # 为验证过程添加进度条
                        val_progress = tqdm(
                            val_loader,
                            desc="Validation",
                            leave=False,
                            dynamic_ncols=True
                        )
                        for vb in val_progress:
                            out = model(vb['iq'], vb['input_ids'], vb['attention_mask'])
                            batch_loss = out.loss.item()
                            val_loss += batch_loss
                            # 更新验证进度条
                            val_progress.set_postfix({'val_loss': f'{batch_loss:.4f}'})
                    val_loss /= len(val_loader)
                    logger.info(f"  → Val Loss: {val_loss:.4f}")
                    
                    # Wandb: 记录验证指标
                    wandb.log({
                        "val/loss": val_loss,
                        "val/global_step": global_step
                    }, step=global_step)

                    # 3) 随机打印 10 个样例，用 "unwrap 后" 的模型 & no_grad
                    sample_v = random.sample(val_indices, k=10)
                    logger.info(f"  → Sample validation IDs: {sample_v}")
                    # 拿到原始模型（去掉 DDP 壳），eval + 关掉grad
                    raw_model = accelerator.unwrap_model(model)
                    raw_model.eval()
                    
                    # 创建wandb表格来保存推理结果
                    inference_data = []
                    
                    with torch.no_grad():
                        for i, vid in enumerate(sample_v):
                            meta = full_dataset.samples[vid]
                            question = meta['question']
                            gt_answer = meta['answer']
                            data = full_dataset[vid]
                            iq   = data['iq'].unsqueeze(0).to(device)
                            ids  = data['input_ids'].unsqueeze(0).to(device)
                            mask = data['attention_mask'].unsqueeze(0).to(device)
                
                            out  = raw_model(iq, ids, mask)
                            pred = out.logits.argmax(dim=-1)[0]
                            B, L, C = iq.shape
                            P = patch_size[0]
                            prefix_len = 1 + (L // P) + 1
                            text_pred = pred[prefix_len:]
                            q_ids = tokenizer(question + tokenizer.eos_token,
                                              return_tensors='pt').input_ids[0]
                            ans_ids = text_pred[q_ids.size(0):]
                            answer = tokenizer.decode(ans_ids,
                                                      skip_special_tokens=True).strip()
                            
                            # 收集推理结果数据
                            inference_data.append([
                                vid,  # 样本ID
                                question,  # 问题
                                answer,  # 预测答案
                                gt_answer,  # 真实答案
                                len(question),  # 问题长度
                                len(answer),  # 预测答案长度
                                len(gt_answer),  # 真实答案长度
                                answer == gt_answer  # 是否完全匹配
                            ])
                            
                            logger.info(f"→ Q:{question!r}\n→ Pred: {answer}\n→ GT: {gt_answer!r}\n")
                    
                    # 创建并记录wandb表格
                    inference_table = wandb.Table(
                        columns=[
                            "Sample_ID", "Question", "Predicted_Answer", "Ground_Truth", 
                            "Question_Length", "Pred_Length", "GT_Length", "Exact_Match"
                        ],
                        data=inference_data
                    )
                    
                    wandb.log({
                        f"inference_results/step_{global_step}": inference_table,
                        "inference_results/exact_match_rate": sum(row[7] for row in inference_data) / len(inference_data)
                    }, step=global_step)
                    model.train()
        # 每个 epoch 写一次平均 loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch)
        
        # Wandb: 记录每个epoch的平均loss
        if accelerator.is_local_main_process:
            wandb.log({
                "train/loss_epoch": avg_epoch_loss,
                "epoch": epoch
            }, step=global_step)
    
    # 关闭总进度条
    if accelerator.is_local_main_process:
        overall_progress.close()
    
    # 关闭 TensorBoard writer
    writer.close()
    
    # 关闭 Wandb
    if accelerator.is_local_main_process:
        wandb.finish()
    
    # 保存模型与 tokenizer
    unwrapped_model = accelerator.unwrap_model(model)
    # 保存 LLM & tokenizer
    unwrapped_model.llm.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 保存 signal_to_hidden MLP & signal_encoder
    torch.save(
        unwrapped_model.signal_to_hidden.state_dict(),
        os.path.join(args.output_dir, "signal_to_hidden.pt")
    )
    torch.save(
        unwrapped_model.signal_encoder.state_dict(),
        os.path.join(args.output_dir, "signal_encoder.pt")
    )
    
    # Wandb: 保存最终模型作为artifact
    if accelerator.is_local_main_process:
        final_artifact = wandb.Artifact("final-model", type="model")
        final_artifact.add_dir(args.output_dir)
        wandb.log_artifact(final_artifact)
        logger.info("Final model saved to wandb as artifact")


if __name__ == '__main__':
    main()