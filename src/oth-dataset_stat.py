"""
Obtain the statistics of Du et al. dataset
"""
import json
from pathlib import Path
from transformers import BertTokenizerFast
from pprint import pprint as print
from utils.logging import logging
from tqdm import tqdm


class Configs:
    txt_data_dir = "../txt_data/preprocessed/"
    txt_data_file_fmt = "para_73k_{split}.json"
    splits = [
        "train", "dev", "test"
    ]
    max_seq_length = 384


if __name__ == "__main__":
    # configs
    cfgs = Configs

    # logging
    logger = logging.getLogger()

    # check input dir
    txt_data_dir = Path(cfgs.txt_data_dir)
    assert txt_data_dir.is_dir()

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    for s in cfgs.splits:
        input_json = txt_data_dir / cfgs.txt_data_file_fmt.format(split=s)
        assert input_json.is_file()
        with open(input_json, "r", encoding="utf-8") as f:
            examples = json.load(f)
        logger.info(f"Num of examples in {s}: {len(examples)}")

        context_set = set()
        num_context_toks = 0
        num_question_toks = 0
        for e in tqdm(examples, desc=s):
            # Context
            context = e["context"]
            if context not in context_set:
                context_toks = tokenizer(context, add_special_tokens=False)["input_ids"]
                num_context_toks += len(context_toks)
                context_set.add(context)

            # Questions
            question = e["question"]
            question_toks = tokenizer(question, add_special_tokens=False)["input_ids"]
            num_question_toks += len(question_toks)

        avg_num_context_toks = num_context_toks / len(context_set)
        avg_num_question_toks = num_question_toks / len(examples)
        logger.info(f"Average number of context tokens in {s}: {avg_num_context_toks:.2f}")
        logger.info(f"Average number of question tokens in {s}: {avg_num_question_toks:.2f}")
