import os
import shutil
import glob
import numpy as np
import torch.nn.functional as F
from datasets import Features, Value, Sequence
from einops import rearrange

import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader

import time
import lmdb
import pickle

import torch
import torch.distributed as dist
import logging

from models import SiTMAEConfig, SiTForMultiTask
from safetensors.torch import load_file
from thop import profile
import inspect
from inspect import Parameter


logger = logging.getLogger(__name__)


def pack_sequence(samples, patch_size, max_seq_len, enable_packing=True):
    max_patches_len = max_seq_len // patch_size 
    sequences = samples

    sample_ids_seq = [] 
    patches_seq = []  
    positions_seq = [] 
    for iq_seq in sequences:
        current_patches = []
        positions = []
        sample_ids = torch.empty((0,), dtype=torch.long)
        for sample_id, sample in enumerate(iq_seq):
            sample_len = sample.shape[-2] 
            num_patches = sample_len // patch_size  
            pos = torch.arange(num_patches)  
            
            iq_patch = rearrange(
                sample, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=patch_size, p2=1
            )
            
            sample_ids = F.pad(sample_ids, (0, iq_patch.shape[-2]), value=sample_id)
            current_patches.append(iq_patch)
            positions.append(pos)
        
        sample_ids_seq.append(sample_ids)
        patches_seq.append(torch.cat(current_patches, dim=0))
        positions_seq.append(torch.cat(positions, dim=0))

    patches = torch.stack(  
        [
            F.pad(patch, (0, 0, 0, max_patches_len - len(patch)), value=0)
            for patch in patches_seq
        ],
        dim=0,
    )

    patch_positions = torch.stack(
        [F.pad(pos, (0, max_patches_len - len(pos)), value=0) for pos in positions_seq],
        dim=0,
    )

    return patches, patch_positions, sample_ids_seq

def transform_downstream(
    batch,
    patch_size,
    use_fs,
    sit_column_name,
    label_column_name,
    reg_column_name,
    norm_method_iq="abs",
    norm_method_reg=None,  
):
    
    if reg_column_name:
        check_columns = [sit_column_name] + [label_column_name] + reg_column_name
    else:
        check_columns = [sit_column_name] + [label_column_name]

    mask = np.ones(len(batch[sit_column_name]), dtype=bool)  
    for col in check_columns:
        if isinstance(batch[col][0], str):  
            col_mask = [(x is not None) and (str(x).strip() != "") for x in data]
            col_mask = np.array(col_mask, dtype=bool)
        else:
            data = np.array(batch[col], dtype=np.float32)
            col_mask = ~np.isnan(data).any(axis=tuple(range(1, data.ndim))) & ~np.isinf(
                data
            ).any(axis=tuple(range(1, data.ndim)))
        mask &= col_mask
        if np.sum(~col_mask) > 0:  
            print(f"Warning: {col} 列数据中存在无效值，已被过滤")

    batch = {k: np.array(v)[mask] for k, v in batch.items()}  
    new_batch_size = len(batch[sit_column_name])
    if new_batch_size == 0:
        print(
            "Warning: 所有数据均含无效值，跳过本批次"
        )  
        return None

    iq_data = batch[sit_column_name]  
    iq_data = normalize_data(batch[sit_column_name], norm_method=norm_method_iq)
    iq_data = np.transpose(np.clip(iq_data, -5.0, 5.0), (0, 2, 1))[
        :, :, :, np.newaxis
    ]  

    cls_token = np.zeros((new_batch_size, 2, patch_size[0], 1))
    
    if use_fs:
        
        fs_token = np.broadcast_to(
            batch["fs"][:, None, None, None], 
            (new_batch_size, 2, patch_size[0], 1),  
        )
        iq_data = np.concatenate(
            (cls_token, fs_token, iq_data), axis=2
        ) 
    else:
        iq_data = np.concatenate((cls_token, iq_data), axis=2)  

    batch[sit_column_name] = iq_data  

    if norm_method_reg is not None and reg_column_name: 
        for col in reg_column_name:
           
            reg_data = batch[col]  
            reg_data = normalize_data(reg_data, norm_method=norm_method_reg)  
            batch[col] = reg_data

    return batch

def normalize_data(data, norm_method="std"):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        raise ValueError("输入数据必须是列表、NumPy 数组或 PyTorch Tensor")

    data = data.astype(float) 
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"Null in normalize_data")  
        return None

    if data.ndim == 1:
        norm_axis = 0
    else:
        norm_axis = tuple(range(1, data.ndim))  

    if norm_method == "std":
        mean = np.mean(data, axis=norm_axis, keepdims=True)
        std = np.std(data, axis=norm_axis, keepdims=True)
        std = np.clip(std, a_min=1e-6, a_max=None)
        normalized_data = (data - mean) / std
    elif norm_method == "abs": 
        norm = np.max(np.abs(data), axis=norm_axis, keepdims=True)
        normalized_data = data / (norm + 1e-6)
    else:
        raise ValueError("不支持的归一化方法，仅支持 'std' 或 'abs'")

    if isinstance(data, list):
        return normalized_data.tolist()
    elif isinstance(data, torch.Tensor):
        return torch.tensor(normalized_data, dtype=data.dtype)
    else:
        return normalized_data


def get_features(parquet_files):
    all_columns = set()
    column_types = {}

    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema_arrow

        for field in schema:
            col_name = field.name
            col_type = field.type
            if col_name not in column_types:
                column_types[col_name] = col_type
            if col_type != pa.null() and column_types[col_name] == pa.null():
                column_types[col_name] = col_type
            all_columns.add(col_name)

    features_dict = {}
    for col in all_columns:
        features_dict[col] = arrow_type_to_datasets_value(column_types[col])
    return Features(features_dict)


def arrow_type_to_datasets_value(arrow_type):
    if arrow_type in (pa.int8(), pa.int32(), pa.int64()):
        return Value("int32")
    elif arrow_type in (pa.float32(), pa.float64()):
        return Value("float32")
    elif arrow_type in (pa.string(), pa.utf8()):
        return Value("string")
    elif arrow_type == pa.bool_():
        return Value("bool")
    elif isinstance(arrow_type, (pa.FixedSizeListType, pa.ListType)):
        return Sequence(Sequence(Value("float32")))
    else:
        return Value("string") 

def std_normalize_tensor(x_tensor, mean, std):
    if x_tensor is not None and mean is not None and std is not None:
        return (x_tensor - mean) / (std + 1e-6)
    else:
        return None

def load_sit_multi_task_model(config, model_name_or_path):
    model = SiTForMultiTask(config)
    pretrained_weights = load_file(
        os.path.join(model_name_or_path, "model.safetensors")
    )
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if key in pretrained_weights:
            if key == "sit.embeddings.position_embeddings":
                pretrained_weights[
                    "sit.embeddings.position_embeddings"
                ] = pretrained_weights["sit.embeddings.position_embeddings"][
                    : model_dict["sit.embeddings.position_embeddings"].shape[0],
                    :,
                ]
            model_dict[key] = pretrained_weights[key]
    model.load_state_dict(model_dict)
    return model
