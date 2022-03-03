from typing import Dict, List, Union
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import logging
import json
logger = logging.getLogger(__name__)


@Predictor.register("crosslingual_dep_parser_predictor")
class CrosslingualDependencyParserPredictor(Predictor):
    """
    Predictor for the crosslingualdependencyparser model.
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader
    ) -> None:
        super().__init__(model, dataset_reader)

    def predict(
        self,
        input_json: Union[Dict, List[Dict]]
    ) -> List[JsonDict]:
        """
        添加predict函数是为predictor作为第三包调用提供函数接口，此类调用不使用allen命令行的predict命令
        """
        if isinstance(input_json, Dict):
            input_json = [input_json]

        return self.predict_batch_json(input_json)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            json_dict['tokens'], json_dict['postags'], json_dict['language'],
        )

    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs, ensure_ascii=False) + "\n"
