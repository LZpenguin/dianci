from typing import Tuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  
        second_dim = other_shape.numel() 

        return torch.gather(
            rearrange(input, "b ... -> b (...)"),  
            0,
            repeat(indices, "z -> z d", d=second_dim),  
        ).reshape(-1, *other_shape)  

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors  
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")  

        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)

        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None

index_first_axis = IndexFirstAxis.apply

class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2

        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)

        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None

index_put_first_axis = IndexPutFirstAxis.apply

def unpad_input_refer(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    hidden_states = cast(torch.Tensor, index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices))
    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

def unpad_input_only_refer(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")

    return index_first_axis(rearranged, indices) 

def unpad_input(
    hidden_states: torch.Tensor,     
    sample_ids_seq: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    
    bs, seq_len = hidden_states.shape[:2]
    device = hidden_states.device

    mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=device)
    for i, ids in enumerate(sample_ids_seq):
        L = ids.numel()
        assert L <= seq_len, f"第 {i} 条序列长度 {L} 超过 seqlen={seq_len}"
        mask[i, :L] = True

    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    hidden_states = cast(torch.Tensor, index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices))

    lengths = []
    for i, ids in enumerate(sample_ids_seq):
        n_sub = int(ids.max().item()) + 1
        for s in range(n_sub):
            cnt = int((ids == s).sum().item())
            lengths.append(cnt)

    lens_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(lens_tensor, dim=0, dtype=torch.int32)
    ], dim=0)

    max_seqlen = int(lens_tensor.max().item())

    return hidden_states, indices, cu_seqlens, max_seqlen

def unpad_input_only(
    hidden_states: torch.Tensor,
    sample_ids_seq: list[torch.Tensor]
) -> torch.Tensor:
    bs, seq_len = hidden_states.shape[:2]
    device = hidden_states.device

    mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=device)
    for i, ids in enumerate(sample_ids_seq):
        L = ids.numel()
        assert L <= seq_len, f"第 {i} 条序列长度 {L} 超过 seqlen={seq_len}"
        mask[i, :L] = True
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")
    return index_first_axis(rearranged, indices)

def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)