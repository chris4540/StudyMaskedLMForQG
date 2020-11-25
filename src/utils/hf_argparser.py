"""
Add more function to parse yaml file
"""
import json
import dataclasses
from pathlib import Path
import transformers
from typing import Tuple, NewType, Any
from utils.logging import logging

# --------------------------------
# self-define typing
DataClass = NewType("DataClass", Any)
# DataClassType = NewType("DataClassType", Any)


def save_cfg_dict(cfg_dict):
    logger = logging.getLogger(__name__ + ".save_cfg_dict")

    output_dir = Path(cfg_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / "arguments.json"

    logger.info(f"The argument output json file is {out_json}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=1, ensure_ascii=False)

def save_args_as_json(arg_instances, out_json=None):
    """
    save dataclasses args into json

    """
    logger = logging.getLogger(__name__ + ".save_args_as_json")

    output_dir = None
    if out_json:
        output_dir = Path(out_json).parent

    # out: the output dictionary dumped into json
    out = dict()
    # out = {k: vars(v) for k, v in arg_dict.items()}
    for arg in arg_instances:
        out[arg.__class__.__name__] = dataclasses.asdict(arg)
        if output_dir:
            continue

        # ------------------------
        # Get output directory
        # ------------------------
        for k, v in dataclasses.asdict(arg).items():
            if k == "output_dir":
                output_dir = Path(v)

    output_dir.mkdir(parents=True, exist_ok=True)
    if out_json is None:
        out_json = output_dir / "arguments.json"
    logger.info(f"The argument output json file is {out_json}")

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


class HfArgumentParser(transformers.HfArgumentParser):
    def parse_yaml_file(self, yaml_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all,
        instead loading a json file and populating the dataclass types.
        """
        try:
            import yaml
        except ImportError:
            raise ValueError("We need the pyyaml package to parse yaml file.")

        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def save_args(self):
        # get the ouput directory
        output_dir = None
        for dtype in self.dataclass_types:
            for k, v in dataclasses.asdict(dtype):
                if k == "output_dir":
                    print(v)
                    output_dir = v
                    break
            if output_dir:
                break
