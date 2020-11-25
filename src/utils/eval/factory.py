"""
Base class of evaluation tools factory
"""

import torch
from torch.utils.data.dataloader import DataLoader
from utils.logging import logging
import dataclasses
from models.bert_qgen import BertForMaskedLM
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.data.datasets.data_collator import DataCollatorForPadding
from utils.eval.arguments import BaseEvalScriptArguments


class BaseEvaluationFactory:
    """
    Base class of factory to give out a correct dataset and decoder according the model
    """
    num_workers: int = 2
    _device: str = "cuda"

    def __init__(self, configs: BaseEvalScriptArguments):
        """
        Parameters
        ----------
        configs:
            A dataclass contains necessary information in your inherit class.

        configs.txt_ds_path: str
            The path to text dataset

        configs.tokenizer_name: str
            The name of the pretrained tokenizer
        """
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
        logger = self.logger
        logger.info("------------ EvaluationFactory configs ------------")
        for k, v in dataclasses.asdict(configs).items():
            logger.info(f"{k:17s} : {str(v)}")

        # -------------------
        # text dataset
        # -------------------
        text_dataset = QnGenHLCtxtDataset.from_cache(configs.txt_ds_path)
        self.text_dataset = text_dataset

        # -------------------
        # tokenizer
        # -------------------
        tokenizer = BertTokenizerWithHLAns.from_pretrained(configs.tokenizer_name)
        self.tokenizer = tokenizer

        # model
        self.model = self._create_model()

    def _create_model(self) -> BertForMaskedLM:
        """
        Create model using the given configs
        """
        model_path = self.configs.model_path
        self.logger.info(f"Loading model: {model_path}")
        model = BertForMaskedLM.from_pretrained(model_path)

        model.to(self.device)
        model.eval()
        return model

    @property
    def device(self) -> str:
        ##############
        # Device
        ##############
        if torch.cuda.is_available() and self._device == "cuda":
            self._device = "cuda"
        else:
            self._device = "cpu"

        return self._device

    def create_dataset(self):
        raise NotImplementedError

    def create_dataloader(self):
        ds = self.create_dataset()

        # make collator
        pad_collator = DataCollatorForPadding(self.tokenizer)

        # make return
        ret = DataLoader(ds, batch_size=self.configs.batch_size, collate_fn=pad_collator,
                         num_workers=self.num_workers, pin_memory=True)

        return ret

    def create_decoder(self):
        raise NotImplementedError

    def create_task_name(self) -> str:
        raise NotImplementedError

    def create_output_filename(self) -> str:
        task_name: str = self.create_task_name()
        ret = f"{task_name}-eval-results.json"
        return ret
