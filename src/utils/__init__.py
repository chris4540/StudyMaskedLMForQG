"""
May need refactoring
"""
from .logging import logging  # noqa: F401, E402
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json


def is_format_string(fmt_string: str) -> bool:
    if isinstance(fmt_string, str):
        ret = re.match(r".*(\{\w*\})+", fmt_string)
        return bool(ret)
    else:
        return False


def save_as_json(json_file, output, indent=1):
    with open(json_file, "w") as f:
        json.dump(output, f, indent=indent)


def load_json(json_file):
    with open(json_file, "r") as f:
        ret = json.load(f)
    return ret


@dataclass
class DataArguments:
    """

    """

    # src_data_folder: str = field(
    #     metadata={"help": "The preprocessed data folder"}
    # )

    txtds_cache_dir: Optional[str] = field(
        default="text_dataset_cache",
        metadata={
            "help": "The text dataset cache folder"
        }
    )

    max_seq_length: Optional[int] = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after WordPiece tokenization."
        }
    )
    doc_stride: Optional[int] = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        }
    )
    max_query_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length."
        }
    )

    # # other configs of this script
    # metric_output: str = "eval_metrics.json"
    # qgen_output_fname: str = "generated_output.json"


def get_chkpts_from(output_dir, checkpoint_prefix="checkpoint", reverse=True):
    if not Path(output_dir).is_dir():
        return None

    glob_chkpts = [
        str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

    # match one by one
    itr_and_pth = list()
    for pth in glob_chkpts:
        # m: the matching return
        m = re.match(f".*{checkpoint_prefix}-([0-9]+)", pth)
        if m and m.groups():
            itr_and_pth.append((int(m.groups()[0]), pth))

    sorted_chkpts = sorted(itr_and_pth, reverse=reverse)
    return sorted_chkpts


def get_lastest_chkpt_from(output_dir, checkpoint_prefix="checkpoint"):

    chkpts = get_chkpts_from(output_dir, checkpoint_prefix=checkpoint_prefix)
    sorted_chkpts = sorted(chkpts, reverse=True)

    if sorted_chkpts:
        # return the first one
        iteration, pth = sorted_chkpts[0]
        ret = pth
        return ret

    return None  # if cannot find any, return None
