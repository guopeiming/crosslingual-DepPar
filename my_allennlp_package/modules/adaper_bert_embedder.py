from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from typing import Optional, Dict, Any, List, Union
from .adapters import Adapter, AdapterBertLayer
import torch.nn as nn
from allennlp.common.checks import ConfigurationError
from transformers import BertModel, ElectraModel, RobertaModel


@TokenEmbedder.register('adapter_bert')
class AdapterBertMismatchedEmbedder(PretrainedTransformerMismatchedEmbedder):

    def __init__(
        self,
        model_name: str,
        adapters: List[Dict[str, Union[Dict[str, Any], List[int]]]] = [{"layers": [i], "params": dict()} for i in range(12)],
        max_length: int = None,
        train_parameters: bool = False,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        load_weights: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        sub_token_mode: Optional[str] = "avg",
    ) -> None:

        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            load_weights=load_weights,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
            sub_token_mode=sub_token_mode
        )

        self.adapters_groups = self.insert_adapters(adapters)

    def insert_adapters(
        self,
        adapters: List[Dict[str, Union[List[int], Dict[str, Any]]]]
    ) -> nn.ModuleList:

        if not isinstance(self._matched_embedder.transformer_model, (BertModel, ElectraModel, RobertaModel)):
            raise ConfigurationError("只支持 *BERT 结构")

        adapters_groups = nn.ModuleList()
        for adapter in adapters:

            adapter_a = Adapter(self.get_output_dim(), **adapter['params'])
            adapter_f = Adapter(self.get_output_dim(), **adapter['params'])

            for i in adapter['layers']:
                layer = self._matched_embedder.transformer_model.encoder.layer[i]
                layer.output = AdapterBertLayer(layer.output, adapter_a)
                layer.attention.output = AdapterBertLayer(layer.attention.output, adapter_f)

            adapters_groups.append(nn.ModuleList([adapter_a, adapter_f]))

        return adapters_groups
