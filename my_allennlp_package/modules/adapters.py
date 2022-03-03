import torch
import torch.nn as nn
from torch.nn.functional import linear
import torch.nn.init as init
from allennlp.nn.activations import Activation
from ..utils.utils import batched_linear
from typing import Union
from transformers.models.bert.modeling_bert import BertOutput, BertSelfOutput


class Adapter(nn.Module):

    def __init__(self, in_feats: int, adapter_size: int = 64, bias: bool = True,
                 activation: str = 'gelu', train_layer_norm: bool = True,
                 dynamic_weights: bool = False):
        super(Adapter, self).__init__()

        self.in_feats = in_feats
        self.adapter_size = adapter_size
        self.bias = bias

        self.weight_down = None
        self.weight_up = None
        self.bias_down = None
        self.bias_up = None
        self.act_fn = Activation.by_name(activation)()
        self.train_layer_norm = train_layer_norm
        self.dynamic_weights = dynamic_weights

        if not dynamic_weights:
            self.weight_down = nn.Parameter(torch.Tensor(adapter_size, in_feats))
            self.weight_up = nn.Parameter(torch.Tensor(in_feats, adapter_size))

            if bias:
                self.bias_down = nn.Parameter(torch.zeros(adapter_size))
                self.bias_up = nn.Parameter(torch.zeros(in_feats))

            self.reset_parameters()

    def forward(self, hidden_states: torch.Tensor):
        linear_func = batched_linear if self.weight_down.dim() == 3 else linear
        x = linear_func(hidden_states, self.weight_down, self.bias_down)
        x = self.act_fn(x)
        x = linear_func(x, self.weight_up, self.bias_up)
        x = x + hidden_states
        return x

    def reset_parameters(self) -> None:
        init.normal_(self.weight_down, std=1e-3)
        init.normal_(self.weight_up, std=1e-3)

    def update_weight(
        self,
        weight_name,
        weight: torch.Tensor,
    ) -> None:
        object.__setattr__(self, weight_name, weight)


class AdapterBertLayer(nn.Module):
    """
    替代 BertOutput 和 BertSelfOutput
    """
    def __init__(self, base: Union[BertOutput, BertSelfOutput], adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter = adapter
        for param in base.LayerNorm.parameters():
            param.requires_grad = adapter.train_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states
