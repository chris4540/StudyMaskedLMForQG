"""
This script is to demo how to generate complete samples from a trained model
"""
import numpy as np
import torch
from utils.logging import logging
from models.bert_qgen import BertForMaskedLM
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.data.datasets.wrapper import uPMLMCondTextGenEvalDatasetWrapper
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from utils.data.datasets.data_collator import DataCollatorForPadding
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from utils.eval.decoding import CausalMLMCondTokenDecoder


class Config:
    """
    A class to place comman configs
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configurations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # change model_path to `hlsqg-p73k-base-out` or `uPMLM-p73k-base-out`
    model_path = "hlsqg-p73k-base-out"
    # model_path = "uPMLM-p73k-base-out"
    tokenizer_name = "bert-base-uncased"
    batch_size = 1
    # If you want to show few results, you can sample the dataset here
    sample_ds = True
    num_samples = 10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, device="cuda"):

        ##############
        # Device
        ##############
        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"
        self._logging()

    def _logging(self):
        logger = logging.getLogger(self.__class__.__name__)

        log_attrs = [
            "model_path",
            "device",
            "batch_size",
            "sample_ds",
            "num_samples",
        ]
        logger.info("---------------------- Configurations ----------------------")
        for a in log_attrs:
            val = getattr(self, a)
            logger.info(f"{a:>12}: {val}")
        logger.info("---------------------- Configurations ----------------------")


class EvaluationFactory:
    """
    Simple factory to give out a correct dataset and decoder according the model
    """

    def __init__(self, configs, text_dataset: QnGenHLCtxtDataset, model, tokenizer):
        self.configs = configs
        self.text_dataset = text_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_decode_len = self.text_dataset.max_query_length

    def create_dataset(self):
        if self.configs.model_path.startswith("hlsqg"):
            # return a simple left-to-right generation dataset
            ret = CausalCondTextGenEvalDatasetWrapper(self.text_dataset, self.tokenizer)
            self.logger.info("Selected CausalCondTextGenEvalDataset")
        elif self.configs.model_path.startswith("uPMLM"):
            # return a unifrom pmlm model generation dataset
            ret = uPMLMCondTextGenEvalDatasetWrapper(
                self.text_dataset,
                self.tokenizer,
                sample_decode_length=True
            )
            self.logger.info("Selected uPMLMCondTextGenEvalDataset")

        # sampling
        if self.configs.sample_ds:
            self.logger.info("Sampling the generation dataset...")
            smpl_idx = np.random.choice(len(ret), size=self.configs.num_samples)
            ret = Subset(ret, smpl_idx)
        return ret

    def create_decoder(self):
        if self.configs.model_path.startswith("hlsqg"):
            decode_length_known = False
        elif self.configs.model_path.startswith("uPMLM"):
            decode_length_known = True

        ret = CausalMLMCondTokenDecoder(
            self.model,
            self.tokenizer,
            max_decode_len=self.max_decode_len,
            decode_length_known=decode_length_known,
            no_repeat_ngram_size=2)
        return ret


if __name__ == "__main__":
    logger = logging.getLogger()
    cfg = Config()
    ##############
    # Load Model
    ##############
    model = BertForMaskedLM.from_pretrained(cfg.model_path)
    model.to(cfg.device)
    model.eval()
    logger.info(f"Loaded Masked Language Model: {cfg.model_path}")

    #########################
    # Load Tokenizer
    #########################
    tokenizer = BertTokenizerWithHLAns.from_pretrained(cfg.tokenizer_name)
    logger.info(f"Loaded Tokenizer: {cfg.tokenizer_name}")

    #########################
    # Load dataset
    #########################
    txtds = QnGenHLCtxtDataset.from_cache("cached_txtds/dev_ds")
    eval_factory = EvaluationFactory(cfg, txtds, model, tokenizer)
    ds = eval_factory.create_dataset()

    #########################
    # Load dataloader
    #########################
    pad_collator = DataCollatorForPadding(tokenizer)
    data_loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        collate_fn=pad_collator,
        num_workers=2,
    )

    #########################################################################
    # Load decoder
    # ----------------------------
    # Notes: This decoder is not the decoder in seq2seq model
    #########################################################################
    decoder = eval_factory.create_decoder()

    for i, inputs in enumerate(data_loader):
        input_ids = inputs["input_ids"]
        question_tok_ids = inputs["question"]
        question_start_pos = inputs["question_start_pos"]
        decoded, score = decoder(inputs)
        id_ = inputs["id"]
        batch_size = question_tok_ids.shape[0]
        for b in range(batch_size):
            assert len(decoded[b]) == 1
            assert len(score[b]) == 1
            q_startpos = question_start_pos[b]
            context = decoder.decode(input_ids[b][:q_startpos])
            ref_question = decoder.decode(question_tok_ids[b])
            hyp_question = decoder.decode(decoded[b][0])
            text = decoder.decode(decoded[b][0])
            hyp_score = score[b][0]
            logger.info(f"id: {id_[b]}")
            logger.info(f"context: {context}")
            logger.info("---------------------------------------------")
            logger.info(f" reference: {ref_question}")
            logger.info(f"hypothesis: {hyp_question}")
            logger.info(f" gen_score: {hyp_score}")
            logger.info("---------------------------------------------")
