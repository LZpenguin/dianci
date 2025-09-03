import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.utils import ModelOutput


@dataclass
class MultiTaskOutput(ImageClassifierOutput):
    loss: torch.Tensor
    logits: dict
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    loss_details: Optional[dict] = None  
    
@dataclass
class SiTMAEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None 
    mae_mask: torch.LongTensor = None 
    ids_restore: torch.LongTensor = None 
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SiTMAEDecoderOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SiTMAEForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    attn_mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None