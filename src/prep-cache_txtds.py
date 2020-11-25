"""
Cache text dataset
"""
import os
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.logging import logging
from pathlib import Path
from models.tokenizer import BertTokenizerWithHLAns


class Config:
    txt_data_dir = "../txt_data/preprocessed/"
    txt_data_file_fmt = "para_73k_{split}.json"
    splits = [
        "train", "dev", "test"
    ]
    cache_dir = "./cached_txtds"
    ds_kwargs = {
        "max_seq_length": 384,
        "max_query_length": 30,
        "doc_stride": 128,
        "processes": 4,
    }


if __name__ == "__main__":
    cfg = Config()

    # check input dir
    txt_data_dir = Path(cfg.txt_data_dir)
    assert txt_data_dir.is_dir()

    # check output dir
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    # tokenizer
    tokenizer = BertTokenizerWithHLAns.from_pretrained("bert-base-uncased")

    for s in cfg.splits:
        input_json = txt_data_dir / cfg.txt_data_file_fmt.format(split=s)
        assert input_json.is_file()
        output_cache = cache_dir / "{split}_ds".format(split=s)
        kwargs = dict(cfg.ds_kwargs)
        kwargs["tokenizer"] = tokenizer
        kwargs["input_json"] = str(input_json)
        if s == "train":
            kwargs["train"] = True
        else:
            kwargs["train"] = False
        txtds = QnGenHLCtxtDataset(**kwargs)

        # cache out
        txtds.cache(output_cache)
