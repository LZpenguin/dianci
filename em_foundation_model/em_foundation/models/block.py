import torch
import math
from transformers.activations import ACT2FN
from .configuration_em import SiTMAEConfig
from torch import nn
from typing import Optional, Tuple, Union
import importlib.metadata
IMPL_USE_FLASH2 = False
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func

    installed_version = importlib.metadata.version("flash_attn")
    if installed_version < "2.5.7":
        raise ImportError("newer version of flash_attn required (>= 2.5.7)")
    IMPL_USE_FLASH2 = True
except ImportError:
    pass

class BaseSiTMAEAttention(nn.Module):
    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads 
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) 
        self.all_head_size = self.num_attention_heads * self.attention_head_size 

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape) 
        return x.permute(0, 2, 1, 3) 

class SiTMAEAttention(BaseSiTMAEAttention):
    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__(config) 

    def forward(
            self, 
            hidden_states, 
            head_mask: Optional[torch.Tensor] = None, 
            output_attentions: bool = False,
            mask: Optional[torch.LongTensor] = None,  
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
       
 
        mixed_query_layer = self.query(hidden_states) 
       
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
       
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)  
            attention_scores = attention_scores.masked_fill(~mask, -1e9)        
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.dropout(self.dense(context_layer))
       
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SiTMAESdpaAttention(BaseSiTMAEAttention):
    
    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob 

    def forward(
            self, 
            hidden_states, 
            head_mask: Optional[torch.Tensor] = None, 
            output_attentions: bool = False,
            mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states)) 
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        mask_for_sdpa = None

        mask_for_sdpa = mask_for_sdpa.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,  
            key_layer,   
            value_layer,  
            attn_mask=mask_for_sdpa, 
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0, 
            is_causal=False,  
            scale=None,      
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  
        context_layer = context_layer.view(new_context_layer_shape)  

        context_layer = self.dropout(self.dense(context_layer))

        return context_layer

class SiTMAEUnpadFlashAttention(nn.Module):
    def __init__(self, config: SiTMAEConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads  
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.qkv_bias)  
        self.attention_dropout = config.attention_probs_dropout_prob 
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,    
        cu_seqlens: torch.Tensor,      
        max_seqlen: int,               
    ):
        bs, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)
        if IMPL_USE_FLASH2:
            
            qkv = qkv.view(-1, 3, self.num_attention_heads, self.attention_head_size)
            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.attention_dropout
                )
                attn = attn.to(orig_dtype)
            else:
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.attention_dropout
                )
            attn = attn.view(bs, dim)
        
        return (self.dropout(self.Wo(attn)), )


SITMAE_ATTENTION_CLASSES = {
    "eager": SiTMAEAttention,
    "sdpa": SiTMAESdpaAttention,
    "fa2": SiTMAEUnpadFlashAttention
}

class SiTMAELayer(nn.Module):
   
    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__()

        self.attention = SITMAE_ATTENTION_CLASSES[config.attention_type](config)
        if config.attention_type == "fa2":
            if IMPL_USE_FLASH2:
                self.use_fa2 = True
            else:
                raise ValueError("No flash_attn required environment")
        else:
            self.use_fa2 = False

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)  
        if isinstance(config.hidden_act, str):  
            self.act_fn = ACT2FN[config.hidden_act]
        else: 
            self.act_fn = config.hidden_act  
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 

    def forward(
            self,
            hidden_states: torch.Tensor,  
            head_mask: Optional[torch.Tensor] = None, 
            output_attentions: bool = False,
            mask: Optional[torch.LongTensor] = None, 

            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        if self.use_fa2:
            self_attention_outputs = self.attention(
                self.norm1(hidden_states), 
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        else:
            self_attention_outputs = self.attention(
                self.norm1(hidden_states),  
                head_mask=head_mask,
                output_attentions=output_attentions,
                mask=mask, 
            )
        attention_output = self_attention_outputs[0]  
        outputs = self_attention_outputs[1:]  

        hidden_states = attention_output + hidden_states
        layer_output = hidden_states + self.dropout(self.w_2(self.act_fn(self.w_1(self.norm2(hidden_states)))))
        outputs = (layer_output,) + outputs
        return outputs