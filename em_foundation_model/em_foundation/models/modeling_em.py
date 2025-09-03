from copy import deepcopy
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss

from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_em import SiTMAEConfig
from .embedder import SiTMAEEmbeddings
from .dataclass import *
from .block import SiTMAELayer
from .padding import *

logger = logging.get_logger(__name__)

class SiTMAEEncoder(nn.Module):

    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [SiTMAELayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor, 
            head_mask: Optional[torch.Tensor] = None, 
            output_attentions: bool = False, 
            output_hidden_states: bool = False,  
            return_dict: bool = True, 
            mask: Optional[torch.LongTensor] = None, 
            
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None
    ) -> Union[tuple, BaseModelOutput]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    mask=mask,  
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, 
                    layer_head_mask, 
                    output_attentions,
                    mask=mask, 
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
           
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,  
            hidden_states=all_hidden_states,  
            attentions=all_self_attentions, 
        )


class SiTMAEPreTrainedModel(PreTrainedModel):

    config_class = SiTMAEConfig
    base_model_prefix = "sit" 
    main_input_name = "patches" 
    supports_gradient_checkpointing = True 
    _supports_sdpa = True  

    def _init_weights(self, module):
        
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SiTMAEModel(SiTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = SiTMAEEmbeddings(config)
        self.encoder = SiTMAEEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        patches: torch.FloatTensor,
        patch_positions: torch.FloatTensor,  
        sample_ids_seq: list[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,  
        head_mask: Optional[torch.FloatTensor] = None,  
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None,  
        return_dict: Optional[bool] = None,  
    ) -> Union[Tuple, SiTMAEModelOutput]:

        bs, seq_len, _ = patches.shape
       
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if patches is None:
            raise ValueError("You have to specify patches")
       
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        patches, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_ids_seq = self.embeddings(
            patches,
            patch_positions=patch_positions,   
            sample_ids_seq = sample_ids_seq,
            noise=noise,
        )
       
        if self.config.attention_type == "fa2":
            patches, indices, cu_seqlens, max_seqlen = unpad_input(patches, unmask_sample_ids_seq)
        else:
            cu_seqlens, max_seqlen = None, None

        encoder_outputs = self.encoder(
            patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask=encoder_attn_mask, 
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if self.config.attention_type == "fa2":
            sequence_output = pad_input(sequence_output, indices, bs, seq_len)

        if not return_dict:
            return (sequence_output, ids_restore) + encoder_outputs[1:]
        
        return SiTMAEModelOutput(
            last_hidden_state=sequence_output,  
            mae_mask=mae_mask,                 
            ids_restore=ids_restore,           
            hidden_states=encoder_outputs.hidden_states,  
            attentions=encoder_outputs.attentions,        
        )

class SiTMAEModelWithoutMask(SiTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = SiTMAEEmbeddings(config)
        self.encoder = SiTMAEEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        samples: torch.Tensor,
        patch_positions: torch.Tensor,
        sample_ids_seq: List[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SiTMAEModelOutput]:
       
        bs, seq_len, _ = samples.shape
       
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if samples is None:
            raise ValueError("You have to specify samples")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        patches, _, _, encoder_attn_mask, unmask_sample_ids_seq = self.embeddings(
            samples,
            patch_positions=patch_positions,  
            sample_ids_seq=sample_ids_seq,
            noise=noise,
            enable_mae_mask=False,
        )

        if self.config.attention_type == "fa2":
            patches, indices, cu_seqlens, max_seqlen = unpad_input(patches, unmask_sample_ids_seq)
        else:
            cu_seqlens, max_seqlen = None, None

        encoder_outputs = self.encoder(
            patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask=encoder_attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )

        sequence_output = encoder_outputs[0]

        if self.config.attention_type == "fa2":
            sequence_output = self.layernorm(sequence_output)       
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return SiTMAEModelOutput(
            last_hidden_state=sequence_output,
            mae_mask=None,
            ids_restore=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiTMAEForClassification(SiTMAEPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.sit = SiTMAEModelWithoutMask(config)
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        self.post_init()

    def forward(
        self,
        patches: torch.FloatTensor,  
        patch_positions: torch.Tensor,
        sample_ids_seq: List[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.sit(
            patches,
            patch_positions,
            sample_ids_seq,
            noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
