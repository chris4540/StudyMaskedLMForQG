"""
A module for metadata write and verification
"""
import json
import dataclasses
from typing import Union
from utils.logging import logging
from os import PathLike


# module logger
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class DatasetMetadata:
    """
    A representation of the metadata for dataset. It makes the cached dataset
    folder readtable
    """
    tokenizer_name: str
    max_seq_length: int
    doc_stride: int
    max_query_length: int
    src_file: str

    def is_match(self, metadata_json: Union[str, PathLike]) -> bool:
        """
        To check if the json file match to this metadata instance.

        Parameters
        ----------
        metadata_json : Union[str, PathLike]
            the input json file

        Returns
        -------
        bool
            True if the input json file is match to this metadata instance and
            vise-versa
        """
        this = dataclasses.asdict(self)
        with open(metadata_json, "r") as f:
            other = json.load(f)

        for k, this_val in this.items():
            other_val = other.get(k, None)
            if this_val != other_val:
                logger.info(f"The metadata '{k}' does not match.")
                logger.info(f"Attribute: {this_val}.")
                logger.info(f"metadata: {other_val}")
                return False
        return True

    def save_as(self, metadata_json: Union[str, PathLike]):
        with open(metadata_json, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        logger.info(f"Saved metadata into {metadata_json}")
