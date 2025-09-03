import math
import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, frequency_embedding_size, max_period=10000):
        half = frequency_embedding_size // 2
        
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimestepEmbedder_trainable(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype
        
        half = frequency_embedding_size // 2
        init_freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.freqs = nn.Parameter(init_freqs, requires_grad=True)

    def timestep_embedding(self, t, frequency_embedding_size):
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimestepEmbedder_triple(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size*3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, frequency_embedding_size, max_period=10000):
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq_k = self.timestep_embedding(t/1000, self.frequency_embedding_size)
        t_freq_m = self.timestep_embedding(t/1000000, self.frequency_embedding_size)
        t_freq_g = self.timestep_embedding(t/1000000000, self.frequency_embedding_size)
        t_emb = self.mlp(torch.cat([t_freq_k, t_freq_m, t_freq_g], dim=-1))
        return t_emb

class SiTMAEEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_seq_len = config.max_seq_len[0] + (16 if config.use_fs else 8)
        max_seq_len = max_seq_len if isinstance(max_seq_len, collections.abc.Iterable) else (max_seq_len, max_seq_len)

        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)

        self.num_patches = max_seq_len[0] // patch_size[0]

        patch_dim = config.num_channels * patch_size[0] * patch_size[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.fs_embedder = TimestepEmbedder(config.hidden_size, dtype=torch.float32)
        self.position_embeddings = nn.Parameter(torch.randn(self.num_patches, config.hidden_size))
        self.projection = nn.Linear(patch_dim, config.hidden_size)

        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        w = self.position_embeddings.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        
        nn.init.normal_(self.fs_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.fs_embedder.mlp[2].weight, std=0.02)

    @staticmethod
    def gen_attn_mask(sample_ids_seq, max_patches_len):
        seq_arange = torch.arange(max_patches_len)
        lengths = torch.tensor([p.shape[0] for p in sample_ids_seq], dtype = torch.int)
        sample_ids_seq = torch.stack([F.pad(p, (0, max_patches_len - len(p)), value=0) for p in sample_ids_seq], dim=0)
        key_pad_mask = (rearrange(seq_arange, 'n -> 1 n') < rearrange(lengths, 'b -> b 1')).to(sample_ids_seq.device) 
        attn_mask = rearrange(sample_ids_seq, 'b i -> b i 1') == rearrange(sample_ids_seq, 'b j -> b 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 j') & rearrange(key_pad_mask, 'b j -> b j 1')
        return attn_mask

    def shuffle_in_sample(self, tensor, start_indices):
        prefix_len = 2 if self.config.use_fs else 1
        tensor_list = tensor.tolist()
        for i in range(tensor.size(0)):
            row = tensor_list[i]
            row_starts = start_indices[i]
            for i in range(len(row_starts) - 1):  
                start = row_starts[i] + prefix_len  
                end = row_starts[i + 1]
                sample = row[start:end]  
                perm = torch.randperm(end - start) 
                shuffled_sample = [sample[idx] for idx in perm.tolist()] 
                row[start:end] = shuffled_sample  

        return torch.tensor(tensor_list).to(tensor.device)

    def random_masking_strict_ratio(self, sample_embeddings, sample_ids_seq):
        device = sample_embeddings.device
        bs, max_patches_len, dim = sample_embeddings.shape

        lengths_list = []
        for p in sample_ids_seq:
            lengths = torch.bincount(p)
            lengths_list.append(lengths) 

 
        start_indices = [] 
        for lengths in lengths_list:
            start_index = torch.cumsum(lengths[:], dim=0)
            start_indices.append(torch.cat((torch.tensor([0], device=device), start_index), dim=0))
         
        simulate_noise = []
        prefix_len = 1 if self.config.use_fs else 0 
        for lengths in lengths_list:
            segments = []
            for l in lengths:
                l = l.item()
                seg = torch.cat([torch.zeros(prefix_len), torch.linspace(0, 1, steps=l-prefix_len)])
                segments.append(seg)
            row = torch.cat(segments)
            if row.numel() < max_patches_len:
                pad = torch.full((max_patches_len - row.numel(),), float('inf'))
                row = torch.cat([row, pad])
            simulate_noise.append(row)
        simulate_noise = torch.stack(simulate_noise).to(sample_embeddings.device)
        simulate_noise = self.shuffle_in_sample(simulate_noise, start_indices)

        num_samples = [len(tensor) for tensor in lengths_list]  
        num_samples = torch.tensor(num_samples, device=device)
        effective_lens = [torch.sum(tensor, dim=0) for tensor in lengths_list]  
        effective_lens = torch.tensor(effective_lens, device=device)  
        len_keep = 2*num_samples + ((effective_lens-2*num_samples)*(1-self.config.mask_ratio)).int()
        len_keep_max, _ = len_keep.max(dim=0)

        ids_shuffle = torch.argsort(simulate_noise, dim=1)
       
        for i in range(bs):
            k = len_keep[i]
            ids_shuffle[i, :k], _ = torch.sort(ids_shuffle[i, :k]) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep_list, sequence_unmasked_list = [], []
        for i in range(bs):
            row_ids_keep = ids_shuffle[i][:len_keep[i]]
            row_unmasked = torch.gather(sample_embeddings[i], dim=0, index=row_ids_keep.unsqueeze(-1).repeat(1, dim))
            ids_keep_list.append(row_ids_keep)
            sequence_unmasked_list.append(row_unmasked)
        
        sequence_unmasked = pad_sequence(sequence_unmasked_list, batch_first=True, padding_value=0)
        
        mae_mask = torch.ones([bs, max_patches_len], device=device) 
        indices = torch.arange(max_patches_len, device=device).unsqueeze(0).repeat(bs, 1)

        mae_mask[indices < len_keep.unsqueeze(1)] = 0 
        mae_mask[indices >= effective_lens.unsqueeze(1)]= 0 
        mae_mask = torch.gather(mae_mask, dim=1, index=ids_restore) 

        if self.config.attention_type == "fa2":
            encoder_attn_mask = None
            unmask_sample_ids_seq = []
            for i in range(bs):
                temp = torch.gather(input=sample_ids_seq[i], dim=0, index=ids_keep_list[i])
                unmask_sample_ids_seq.append(temp)
        else:
            attn_mask = self.gen_attn_mask(sample_ids_seq, max_patches_len)
            encoder_attn_mask = torch.zeros(
                                    (bs, len_keep_max, len_keep_max), 
                                    dtype=torch.bool, 
                                    device=device
                                )
            for i in range(bs):
                sub_mask = attn_mask[i][ids_keep_list[i]][:, ids_keep_list[i]] 
                encoder_attn_mask[i][:len_keep[i], :len_keep[i]] = sub_mask.bool()  
            unmask_sample_ids_seq = None
        
        return sequence_unmasked, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_ids_seq

    def random_masking_row(self, sample_embeddings, sample_ids_seq):
        device = sample_embeddings.device
        bs, max_patches_len, dim = sample_embeddings.shape
        lengths_list = []
        for p in sample_ids_seq:
            lengths = torch.bincount(p)
            lengths_list.append(lengths)
        start_indices = [] 
        for lengths in lengths_list:
            start_index = torch.cumsum(lengths[:-1], dim=0) 
            start_indices.append(torch.cat((torch.tensor([0], device=device), start_index), dim=0))
    
        num_samples = [len(tensor) for tensor in lengths_list]
        num_samples = torch.tensor(num_samples, device=device)
        effective_lens = [torch.sum(tensor, dim=0) for tensor in lengths_list]
        effective_lens = torch.tensor(effective_lens, device=device)
        len_keep = 2*num_samples + ((effective_lens-2*num_samples)*(1-self.config.mask_ratio)).int()
        len_keep_max, _ = len_keep.max(dim=0)
  
        noise = torch.rand(bs, max_patches_len, device=device) 
        cls_indices = torch.cat([
            torch.stack([torch.full_like(t, i), t], dim=1)
            for i, t in enumerate(start_indices)
        ], dim=0)
        noise[cls_indices[:, 0], cls_indices[:, 1]] = 0
        if self.config.use_fs:
            next_col = cls_indices[:, 1] + 1
            noise[cls_indices[:, 0], next_col] = 0
        mask = torch.arange(noise.size(1), device=noise.device).unsqueeze(0) >= effective_lens.unsqueeze(1)
        noise = noise.masked_fill(mask, float('inf'))
  
        ids_shuffle = torch.argsort(noise, dim=1)
        for i in range(bs):
            k = len_keep[i]
            ids_shuffle[i, :k], _ = torch.sort(ids_shuffle[i, :k])
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep_list, sequence_unmasked_list = [], []
        for i in range(len(sample_embeddings)):
            row_ids_keep = ids_shuffle[i][:len_keep[i]]
            row_unmasked = torch.gather(sample_embeddings[i], dim=0, index=row_ids_keep.unsqueeze(-1).repeat(1, dim))
            ids_keep_list.append(row_ids_keep)
            sequence_unmasked_list.append(row_unmasked)
        
        sequence_unmasked = pad_sequence(sequence_unmasked_list, batch_first=True, padding_value=0)
        
        mae_mask = torch.ones([bs, max_patches_len], device=device)
        indices = torch.arange(max_patches_len, device=device).unsqueeze(0).repeat(bs, 1)
        mae_mask[indices < len_keep.unsqueeze(1)] = 0
       
        mae_mask[indices >= effective_lens.unsqueeze(1)]= 0 
        mae_mask = torch.gather(mae_mask, dim=1, index=ids_restore)

       
        if self.config.attention_type == "fa2":
            encoder_attn_mask = None
            unmask_sample_ids_seq = []
            for i in range(bs):
                temp = torch.gather(input=sample_ids_seq[i], dim=0, index=ids_keep_list[i])
                unmask_sample_ids_seq.append(temp)
        else:
            attn_mask = self.gen_attn_mask(sample_ids_seq, max_patches_len)
            encoder_attn_mask = torch.zeros(
                                    (bs, len_keep_max, len_keep_max), 
                                    dtype=torch.bool,
                                    device=sample_embeddings.device
                                )
            for i in range(bs):
                sub_mask = attn_mask[i][ids_keep_list[i]][:, ids_keep_list[i]]
                encoder_attn_mask[i][:len_keep[i], :len_keep[i]] = sub_mask.bool()

        return sequence_unmasked, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_ids_seq

    def forward(self,
                patches: torch.Tensor,          
                patch_positions: torch.Tensor,  
                sample_ids_seq: List[torch.Tensor],
                noise: Optional[torch.Tensor] = None,
                enable_mae_mask: bool = True,
                ):
  
        if self.config.use_fs:
            fs_mask = (patch_positions == 1)
            selected_patch = patches[fs_mask] 
            fs = selected_patch[:, 0] 
        patch_embeddings = self.projection(patches)  
        bs, max_patches_len, hidden_size = patch_embeddings.shape

        cls_mask = (patch_positions == 0).unsqueeze(-1) 

        cls_token_expanded = self.cls_token.expand(bs, max_patches_len, hidden_size)
        
        patch_embeddings = torch.where(cls_mask, cls_token_expanded, patch_embeddings)

        if self.config.use_fs:
            fs_embeddings = self.fs_embedder(fs).to(patch_embeddings.dtype)
            patch_embeddings[fs_mask] = fs_embeddings 
       
        patch_embeddings = patch_embeddings + self.position_embeddings[patch_positions]

        if enable_mae_mask:
            embeddings, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_eds_seq = self.random_masking_strict_ratio(
                patch_embeddings, sample_ids_seq
                )
        else:
            embeddings = patch_embeddings
            ids_restore = None
            mae_mask = None
            if self.config.attention_type == "fa2":
                unmask_sample_eds_seq = sample_ids_seq  
                encoder_attn_mask = None
            else:
                attn_mask = self.gen_attn_mask(sample_ids_seq, max_patches_len)
                encoder_attn_mask = attn_mask  
                unmask_sample_eds_seq = None

        return embeddings, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_eds_seq