"""
Script to prepare the text dataset of triplet and a triplet is of:
    - context
    - question
    - answer

No NLP in the procedures of this script.

Prerequisite
--------------
Two files from SQuAD v1.1:
    squad/train-v1.1.json
    squad/dev-v1.1.json

Outputs
--------------
    para_73k_dev.json
    para_73k_test.json
    para_73k_train.json
    patched_train-v1.1.json
    patched_dev-v1.1.json

TODO
----------------
Fix this script

"""
from utils.preprocessing.text_dataset import SQuADV1TextDataset
from utils.preprocessing.patching import PatchedSQuADTextDataset
from utils.preprocessing.squad import ParagraphSQuAD73kSplitsBuilder
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class Configs:
    data_dir: str = "../txt_data/squad"
    blocklist_json: str = "../txt_data/patch/json/blocklist.json"
    patched_question_json: str = "../txt_data/patch/json/patch_questions.json"
    output_dir: str = "./prep_txt_ds"
    split_info_dir: str = "../txt_data/split_info"


if __name__ == "__main__":
    cfg = Configs()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Work directory: {output_dir}")
    for s in ["train", "dev"]:
        ds = SQuADV1TextDataset(cfg.data_dir, split=s)
        patched_ds = PatchedSQuADTextDataset(
            ds, cfg.blocklist_json, cfg.patched_question_json)
        # -------------------------------------------------
        filepath = output_dir / f"patched_{s}-v1.1.json"
        patched_ds.to_json(filepath)
        logging.info(f"Wrote patched data into {filepath}")
        b = ParagraphSQuAD73kSplitsBuilder(cfg.split_info_dir, output_dir, patched_ds)
        b.save_examples()
