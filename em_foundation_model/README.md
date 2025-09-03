
# EM Foundation Model

**项目文件架构**

.
├── 9G4B/                   # 9G4B大语言模型，建议从原始链接下载并替换
├── em_foundation/         # 电磁信号基础大模型
├── run_train.sh           # 信号多模态一键训练脚本,调用 train_mllm.py
└── train_mllm.py          # 信号多模态训练 / 评估主程序

**run_train.sh 和 train_mllm.py为多模态示例代码，仅提供基本的实现思路，可根据需要做修改。**

---

## 1. 架构

#### 1.1 EM Foundation Model可参阅下面文章

https://arxiv.org/abs/2508.00274

1. **原始数据**：`N × 2` IQ 序列。  
2. **Patch → Token**： 将原始 IQ 流按 8 个采样点切片，送入EM Foundation Model获取特征表示。

#### 1.2 多模态框架示例介绍

本项目提供一个多模态框架示例，可将九格4B大语言模型与EM Foundation Model多模态适配连接，仅提供基本的实现思路。

| 组件 | 目录 | 作用 |
|---|---|---|
| **电磁信号基础模型** | `em_foundation/models/` | 电磁信号基础模型，获取IQ信号序列的特征表示。 |
| **大语言模型（LLM）** | `9G4B/` | 九格4B大语言模型，作为多模态交互枢纽和推理核心，详细文档请参阅 [**FM9G-4B**](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/quick_start_4_7_70/inference_4b.md)，请从该链接下载模型并替换 |
| **多模态框架示例** | train_mllm.py | 1) 通过 `nn.Linear(d_em→d_llm)` 对齐到 LLM 维度；2) 在 token 序列前部插入特殊标记 `<IQ_START> … IQ_EMBEDS … <IQ_END>`；3) 通过 信号`inputs_embeds` 与文本 `input_embeds` 拼接后送入 LLM，实现分析推理。 |

多模态框架整体逻辑如下：
```text
IQ(N × 2) ─► EM Foundation Model ─► K×d_em ─► Linear(d_em→d_llm) ───┐
                                                          			├─►  [<IQ_START>  
text tokens ────────────────────────────────────────────────────────┘

IQ_EMBEDS <IQ_END>  text tokens] ─► LLM ─► loss
```

---

## 2. 模型下载

EM Foundation Model模型下载链接如下，下载完毕后替换路径em_foundation/weight内容即可。

通过网盘分享的文件：weight
链接: https://pan.baidu.com/s/1YFBuisvhR4fVYexjbfKFdQ?pwd=tnz7 提取码: tnz7 


---

## 3. 环境配置

### 3.1 创建 Conda 环境

首先，使用 Conda 创建一个新的虚拟环境，并指定 Python 版本：

```bash
conda create -n em python=3.10.16
```

激活新创建的环境：

```bash
conda activate em
```

### 3.2 安装 PyTorch

安装与 CUDA 12.1 兼容的 PyTorch 及其相关库：

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3.3 安装相关依赖库

请依次执行以下命令进行安装：

```bash
pip install transformers==4.44.0
pip install datamodel-code-generator
pip install accelerate
pip install jsonschema
pip install pytrie
pip install sentencepiece
pip install protobuf
pip install h5py
pip install peft
pip install einops
pip install tensorboard
```


---

## 4. 数据格式

数据集请在相关页面下载，格式说明如下。

信号数据以.h5格式保存，相关情况如下表，详细label描述请参阅数据集文件夹Readme。

| Key | Shape / Type | 描述 |
|-----|--------------|------|
| `IQ_data` | `float32[Num, 1024, 2]` | 原始 I/Q 序列 |
| `label`   | `float32[Num, 1]` | 分类标签 |

多模态数据以.json格式保存，示例如下：

```json
[
    {
        "id": 0,
        "question": "请依据给出的IQ信号内容，选出最符合的调制类型（四选一）。(A) GFSK (B) FM (C) MSK (D) 2FSK",
        "answer": "正确答案是：(B) FM",
        "iq_index_in_h5": 0,
        "h5_path": "modulation.h5"
    }
]
```

---


## 5. 多模态配置参数

示例代码train_mllm.py的配置说明如下，具体实现不局限于此代码。

| 参数 | 含义 |
|---|---|
| `--task-dir / --task-files` | 信号/多模态数据目录 |
| `--signal-encoder-path` | EM Foundation Model文件夹路径 |
| `--llm-model-path` | LLM（9G4B) 文件夹路径 |
| `--output-dir` | 模型与日志输出目录 |
| `--batch-size` | 每 GPU 训练样本数 |
| `--epochs` | 训练轮数 |
| `--lr` | 学习率 |
| `--max-length` | LLM 最大 token 数 |
| `--freeze-llm` | 冻结 LLM 参数 |
| `--freeze-signal-encoder` | 冻结 Signal Encoder |
| `--use-lora` | 是否LoRA 微调 |

---

