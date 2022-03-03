from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from typing import Optional, Dict, Any, List, Union
from .adaper_bert_embedder import AdapterBertMismatchedEmbedder
import torch.nn as nn
import torch
import torch.nn.init as init
from allennlp.data import Vocabulary


@TokenEmbedder.register('pgn_adapter_bert')
class PGNAdapterBertMismatchedEmbedder(AdapterBertMismatchedEmbedder):

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        pgn_emb_num: Union[int, str],
        adapters: List[Dict[str, Union[Dict[str, Any], List[int]]]] = [{"layers": [i], "params": dict()} for i in range(12)],
        pgn_emb_dim: int = 8,
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

        for adapter in adapters:
            adapter['params']['dynamic_weights'] = True

        super().__init__(
            model_name,
            adapters,
            max_length,
            train_parameters,
            last_layer_only,
            override_weights_file,
            override_weights_strip_prefix,
            load_weights,
            gradient_checkpointing,
            tokenizer_kwargs,
            transformer_kwargs,
            sub_token_mode,
        )

        if isinstance(pgn_emb_num, str):
            if pgn_emb_num not in vocab.get_namespaces():
                raise ConfigurationError(f"{pgn_emb_num} is not in vocab namespace.")
            pgn_emb_num = vocab.get_vocab_size(pgn_emb_num)
        self.embedding = nn.Embedding(pgn_emb_num, pgn_emb_dim)
        self.pgn_groups = self.pgn_groups_init(pgn_emb_dim)

    def forward(
        self,
        pgn_ids: torch.LongTensor,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        self.parameters_generation(pgn_ids)
        return super().forward(token_ids, mask, offsets, wordpiece_mask, type_ids, segment_concat_mask)

    def parameters_generation(self, pgn_ids: torch.LongTensor) -> None:
        e = self.embedding(pgn_ids)  # [batch, dim]

        for adapters_g, pgn_g in zip(self.adapters_groups, self.pgn_groups):
            for adapter, pgn in zip(adapters_g, pgn_g):
                for weight_name, pgn_weight in pgn.items():
                    ALPHA = 'abcdefg'
                    dims = ALPHA[:pgn_weight.dim()-1]
                    adapter_weight = torch.einsum(f'{dims}k,nk->n{dims}', pgn_weight, e)
                    adapter.update_weight(weight_name, adapter_weight)

    def pgn_groups_init(self, pgn_emb_dim: int) -> nn.ModuleList:
        pgn_groups = nn.ModuleList()
        for adapters_group in self.adapters_groups:
            pgn_group = nn.ModuleList()
            for adapter in adapters_group:
                weights_dict = {
                    'weight_down': nn.Parameter(init.normal_(torch.Tensor(
                        adapter.adapter_size, adapter.in_feats, pgn_emb_dim), std=1e-3)),
                    'weight_up': nn.Parameter(init.normal_(torch.Tensor(
                        adapter.in_feats, adapter.adapter_size, pgn_emb_dim), std=1e-3))
                }
                if adapter.bias:
                    weights_dict['bias_down'] = nn.Parameter(torch.zeros(
                        adapter.adapter_size, pgn_emb_dim))
                    weights_dict['bias_up'] = nn.Parameter(torch.zeros(
                        adapter.in_feats, pgn_emb_dim))
                pgn_group.append(nn.ParameterDict(weights_dict))
            assert len(pgn_group) == len(adapters_group)
            pgn_groups.append(pgn_group)
        assert len(pgn_groups) == len(self.adapters_groups)
        return pgn_groups
