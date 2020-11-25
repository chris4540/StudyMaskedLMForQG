"""
Ad-hoc script for preparing test case data
"""

import json
from pathlib import Path
import random


class Config:
    """
    Configurations
    """
    data_folder = "../../txt_data/preprocessed"
    file_pattern = "sentence_73k_{split}.json"
    splits = ["train", "dev", "test"]
    nsamples = {
        "train": 2,
        "dev": 1,
        "test": 1
    }
    random_seed = 100
    output_dir = "./mock_data/squad_sent_73k"


if __name__ == "__main__":
    cfg = Config()

    random.seed(cfg.random_seed)
    script_dir = Path(__file__).parent
    data_folder = script_dir / Path(cfg.data_folder)
    output_dir = script_dir / Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for s in cfg.splits:
        json_file = data_folder / cfg.file_pattern.format(split=s)
        assert json_file.is_file()

        print(f"Handling {s} data.")
        print(f"Reading {json_file.as_posix()}")
        with json_file.open("r") as f:
            examples = json.load(f)

        n_samples = cfg.nsamples[s]
        samples = random.sample(examples, k=n_samples)
        assert len(samples) == n_samples
        output_json = output_dir / cfg.file_pattern.format(split=s)
        total = len(examples)
        print(f"Writing file: {output_json.as_posix()}. ")
        print(f"Sampled {n_samples} out of {total} examples.")
        with output_json.open("w") as f:
            json.dump(samples, f, indent=1)
