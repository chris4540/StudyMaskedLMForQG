"""
                            Script Documentation

This script is to generate 30 sets of questions with context for human evalaution.

Three models / generation mehtods are used:
    - hlsqq + left-to-right decoding
    - uPMLM + left-to-right decoding
    - uPMLM + random decoing

Steps:
    - generate `hlsqq + left-to-right decoding` first
    - generate `uPMLM + left-to-right decoding` and `uPMLM + random decoing`
      together
"""
import json
import yaml
from tqdm import tqdm
from utils.logging import logging
from models.tokenizer import BertTokenizerWithHLAns
from models.bert_qgen import BertForMaskedLM
from torch.utils.data.dataloader import DataLoader
from utils.data.datasets.wrapper import uPMLMCondTextGenEvalDatasetWrapper
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.data.datasets.data_collator import DataCollatorForPadding
from utils.eval.decoding import CausalMLMBSCondTokenDecoder
from utils.eval.decoding import uPMLMBSCondTokenDecoder
from utils.eval.decoding import BaseBeamSearchTokenDecoder
from typing import Dict, Set
from torch.utils.data import Subset
from utils.data.datasets.squad import highlight_answer


class Configs:
    test_file = "../txt_data/preprocessed/para_73k_test.json"
    selected_idx_yaml = "../txt_data/evaluation/selected_idxs.yaml"
    model_paths = {
        "casual": "hlsqg-p73k-base-out",
        "u-PMLM": "uPMLM-p73k-base-out"
    }
    output_cache_fname = "../txt_data/evaluation/bs-results-for-qtn.json"


class Factory:
    """
    Simple factory to give out a correct dataset and decoder according the model
    """

    def __init__(self, configs, text_dataset: QnGenHLCtxtDataset, model, tokenizer):
        self.configs = configs
        self.text_dataset = text_dataset
        self.model = None
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_decode_len = self.text_dataset.max_query_length

        self.selected_ids = self._get_selected_ids()
        self._make_id_to_text_dataset_id()

    def _get_selected_ids(self) -> Set[str]:
        file_ = self.configs.selected_idx_yaml
        self.logger.info(f"Loading {file_} ......")
        with open(file_, "r", encoding="utf-8") as f:
            ret = yaml.load(f, Loader=yaml.FullLoader)
        return ret

    def _make_id_to_text_dataset_id(self):
        _id_to_idx: Dict[str, int] = dict()
        cnt = 0
        for i, d in enumerate(self.text_dataset):
            id_ = d["id"]
            if id_ in self.selected_ids:
                _id_to_idx[id_] = i
                cnt += 0
            if cnt == len(self.selected_ids):
                break

        self._id_to_idx = _id_to_idx

    def create_model(self, type_: str) -> BertForMaskedLM:
        model_path = self.configs.model_paths[type_]
        self.logger.info(f"Loading model: {model_path}")
        model = BertForMaskedLM.from_pretrained(model_path)
        return model

    def create_causal_dataset(self):
        ds = CausalCondTextGenEvalDatasetWrapper(self.text_dataset, self.tokenizer)
        self.logger.info("Subseting the casual dataset...")
        # get sample index from _id_to_idx
        smpl_idx = [v for v in self._id_to_idx.values()]
        ret = Subset(ds, smpl_idx)
        return ret

    def create_uPMLM_dataset(self):
        ds = uPMLMCondTextGenEvalDatasetWrapper(
            self.text_dataset, self.tokenizer,
            sample_decode_length=True,
            sample_params={"poisson": {"lambda": 12.22, "min": 1}})
        self.logger.info("Subseting the u-PMLM dataset...")
        # get sample index from _id_to_idx
        smpl_idx = [v for v in self._id_to_idx.values()]
        ret = Subset(ds, smpl_idx)
        return ret

    def create_dataloader(self, type_: str) -> DataLoader:
        possible_types = ["casual", "u-PMLM"]
        if type_ not in possible_types:
            _msg = f"The input type should be one of {possible_types}"
            raise ValueError(_msg)

        # make dataset
        if type_ == "casual":
            ds = self.create_causal_dataset()
        elif type_ == "u-PMLM":
            ds = self.create_uPMLM_dataset()
        else:
            ds = None

        # make collator
        pad_collator = DataCollatorForPadding(self.tokenizer)

        # make return
        ret = DataLoader(ds, batch_size=1, collate_fn=pad_collator,
                         num_workers=2)

        return ret

    def create_decoder(self, type_: str, model: BertForMaskedLM, decode_length_known=False) -> BaseBeamSearchTokenDecoder:
        ret: BaseBeamSearchTokenDecoder

        if type_ == "casual":
            ret = CausalMLMBSCondTokenDecoder(
                model, self.tokenizer,
                no_repeat_ngram_size=2,
                decode_length_known=decode_length_known,
                num_return_sequences=1)
        elif type_ == "random_order":
            ret = uPMLMBSCondTokenDecoder(
                model, self.tokenizer,
                no_repeat_ngram_size=2,
                decode_length_known=decode_length_known,
                num_return_sequences=1)
        else:
            _msg = "The type of decoder is either `casual` or random"
            raise ValueError(_msg)

        return ret

    def make_output_cache(self) -> Dict[str, Dict]:
        """
        Make the `output_cache` with content and reference question inside inside.
        """
        ret: Dict[str, Dict] = {k: dict() for k in self.selected_ids}
        # load test file
        with open(self.configs.test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for example in data:
            id_ = example["id"]
            if id_ not in self.selected_ids:
                continue

            # highlight the example
            hl_example = highlight_answer(example)
            _fact = {
                "context": hl_example["highlighted_context"],
                "title": example["title"],
                "question": example["question"],
                "answer_text": example["answer_text"]
            }
            ret[id_]["fact"] = _fact
        return ret


def create_cache(model, decoder, inputs):
    batch_size = inputs["question"].shape[0]
    assert batch_size == 1

    # the reference questions
    question_tok_ids = inputs["question"][0]
    ref_question = decoder.decode(question_tok_ids)
    decodeds, scores = decoder(inputs)
    # transform them to text
    hyp_question = decoder.decode(decodeds[0][0])
    score = scores[0][0]

    # add results to cache
    cache = {
        "reference": ref_question,
        "hypothesis": hyp_question,
        "score": score
    }
    return cache


def save_output_caches(output_caches):
    with open(configs.output_cache_fname, "w", encoding="utf-8") as f:
        json.dump(output_caches, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    # logger
    logger = logging.getLogger()

    # configs
    configs = Configs()

    # tokenizer
    tokenizer = BertTokenizerWithHLAns.from_pretrained("bert-base-uncased")

    text_ds = QnGenHLCtxtDataset.from_cache("cached_txtds/test_ds")

    # factory
    factory = Factory(
        configs=configs,
        text_dataset=text_ds,
        model=None,
        tokenizer=tokenizer
    )
    # output cache
    output_caches = factory.make_output_cache()
    # ----------------------------------------------------------------
    # method 1: sequential trained model + sequential decoding
    method_name = "hlsqg_casual"
    model = factory.create_model("casual")
    dataloader = factory.create_dataloader("casual")
    decoder = factory.create_decoder("casual", model, decode_length_known=False)

    for inputs in tqdm(dataloader, desc=method_name):
        batch_size = len(inputs["id"])
        assert batch_size == 1
        # the Q&A id
        id_ = inputs["id"][0]
        cache = create_cache(model, decoder, inputs)
        output_caches[id_][method_name] = cache

    # save the cache
    save_output_caches(output_caches)

    # ----------------------------------------------------------------
    # method 2
    method_name = "u-PMLM_casual"
    model = factory.create_model("u-PMLM")
    dataloader = factory.create_dataloader("u-PMLM")
    decoder = factory.create_decoder("casual", model, decode_length_known=True)

    for inputs in tqdm(dataloader, desc=method_name):
        batch_size = len(inputs["id"])
        assert batch_size == 1
        # the Q&A id
        id_ = inputs["id"][0]
        cache = create_cache(model, decoder, inputs)
        output_caches[id_][method_name] = cache

    # save the cache
    save_output_caches(output_caches)

    # --------------------------------------------------------
    # method 3
    # use the last model and dataloader
    method_name = "u-PMLM_random"
    decoder = factory.create_decoder("random_order", model, decode_length_known=True)

    for inputs in tqdm(dataloader, desc=method_name):
        batch_size = len(inputs["id"])
        assert batch_size == 1
        # the Q&A id
        id_ = inputs["id"][0]
        cache = create_cache(model, decoder, inputs)
        output_caches[id_][method_name] = cache

    # save the cache
    save_output_caches(output_caches)
