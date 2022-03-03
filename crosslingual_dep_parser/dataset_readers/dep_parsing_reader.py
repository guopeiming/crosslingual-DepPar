from typing import Dict, Iterator, List, Union
import logging
import torch
import json

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, SequenceLabelField, LabelField, TensorField
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)


def data_generator_from_file(file_path: str) -> Iterator[Dict[str, Union[List[Union[str, int, float]], str]]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            yield json.loads(line)


@DatasetReader.register("crosslingual_dep_parsing_reader")
class CrosslingualDepParsingDatasetReader(DatasetReader):

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str):
        logger.info("Reading semantic dependency parsing data from: %s", file_path)
        for annotated_sentence in data_generator_from_file(file_path):
            yield self.text_to_instance(**annotated_sentence)

    def text_to_instance(
        self,
        tokens: List[str],
        postags: List[str],
        language: str,
        heads: List[int] = None,
        deprels: List[str] = None,
        confidences: List[float] = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}

        token_field = TextField(self._tokenizer.tokenize(' '.join(tokens)), self._token_indexers)
        fields["tokens"] = token_field
        fields["postags"] = SequenceLabelField(postags, token_field, label_namespace="pos")
        fields["language"] = LabelField(language, label_namespace='languages_labels')
        fields["metadata"] = MetadataField({"tokens": tokens, "postags": postags, "language": language})

        if heads is not None:
            fields["heads"] = SequenceLabelField(heads, token_field, label_namespace='heads_tags')
        if deprels is not None:
            fields['deprels'] = SequenceLabelField(deprels, token_field, label_namespace='deprels_tags')
        if confidences is not None:
            fields['confidences'] = TensorField(torch.tensor(confidences), padding_value=0.)

        return Instance(fields)
