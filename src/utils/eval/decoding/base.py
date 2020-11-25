import torch
from torch import Tensor
from typing import Dict
from copy import copy
from utils.logging import logging


class BaseTokenDecoder:
    """
    The base class for text generation decoding procedure
    """

    def __init__(self, model=None, tokenizer=None,
                 max_decode_len=50,
                 no_repeat_ngram_size=2,
                 num_return_sequences=1,
                 length_penalty=None,
                 decode_length_known=False):

        if model:
            self._model = model
        if tokenizer:
            self._tokenizer = tokenizer

        self.decode_length_known = decode_length_known
        self.max_decode_len = max_decode_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_return_sequences = num_return_sequences
        if length_penalty is None:
            self.length_penalty = 1
        else:
            self.length_penalty = length_penalty

        # logging for debug
        self.logger = logging.getLogger(self.__class__.__name__)

        # logging
        self._logging()

    def _logging(self):
        log_attrs = [
            "length_penalty",
            "num_return_sequences",
            "no_repeat_ngram_size",
            "max_decode_len",
            "decode_length_known",
        ]
        logger = self.logger
        logger.info("----------------- Decoder Attributes -----------------")
        for a in log_attrs:
            val = getattr(self, a)
            logger.info(f"{a:>12}: {val}")
        logger.info("----------------- Decoder Attributes -----------------")


    @property
    def name(self):
        ret = self.__class__.__name__
        return ret

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, val):
        self._tokenizer = val

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert self.model is not None
        assert self.tokenizer is not None
        return self.forward(*args, **kwargs)

    def decode(self, token_ids):
        ret = self.tokenizer.decode(token_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
        return ret

    def convert_ids_to_tokens(self, token_ids):
        tokenizer = self.tokenizer
        assert tokenizer is not None
        ret = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=True)
        return ret

    @torch.no_grad()
    def _clone_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        clone the input to avoid modificaiton
        TODO: Test this function
        """
        ret = dict()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.clone()
            else:
                ret[k] = copy(v)
            # assert id(v) != id(ret[k])
        return ret
