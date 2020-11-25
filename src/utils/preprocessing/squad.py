from utils.preprocessing.text_dataset import SQuADTextDataset
from utils.preprocessing.text_dataset import SquadExample
from utils.logging import logging
from pathlib import Path
from typing import List, Dict, Union
from pprint import pformat
import json
from os import PathLike

PathType = Union[PathLike, str]


class MapTitleToSplit:
    """
    Load the doclists to instaniate this object

    Reference
    ---------
    Paper link: https://arxiv.org/abs/1705.00106

    ```bible
    @inproceedings{du2017learning,
        title={Learning to Ask: Neural Question Generation for Reading Comprehension},
        author={Du, Xinya and Shao, Junru and Cardie, Claire},
        booktitle={Association for Computational Linguistics (ACL)},
        year={2017}
    }
    ```
    """

    splits: list = ["train", "dev", "test"]
    doclist_file_fmt: str = "doclist-{split}.txt"
    title_to_ds: Dict[str, str]

    def __init__(self, doclist_dir):
        logger = logging.getLogger(self.__class__.__name__)
        self.logger = logging

        # the actual mapping from title name to split
        self._title_to_split = dict()

        # load the doclists
        doclist_folder = Path(doclist_dir)
        logger.info(f"The doclist folder is: {doclist_folder}")
        for s in self.splits:
            file = doclist_folder / self.doclist_file_fmt.format(split=s)
            logger.info(f"Loading the doclist: {file}")
            with file.open('r', encoding='utf-8') as f:
                for line in f:
                    self._title_to_split[line.rstrip("\n")] = s

    def __getitem__(self, key: str) -> str:
        return self._title_to_split[key]

    def __repr__(self):
        return pformat(self._title_to_split)


class SQuAD73kSplitsBuilder:
    output_json_fmt: str = ""
    split_info_subdir: str = ""

    def __init__(self, split_info_dir: PathType, output_dir: PathType, ds: SQuADTextDataset):
        if not self.output_json_fmt or not self.split_info_subdir:
            raise ValueError(
                "You should inherit this class and define `output_json_fmt` and `split_info_subdir`")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Loading doclists to get title-to-split mapping")

        self.output_dir = output_dir

        # build doclist
        doclist_dir = Path(split_info_dir) / self.split_info_subdir
        self.title_to_split = MapTitleToSplit(doclist_dir)

        #
        self.splits: List[str] = self.title_to_split.splits

        #
        self.split_to_examples: Dict[str, List[SquadExample]]
        self.split_to_examples = {s: list() for s in self.splits}

        for d in ds:
            s = self.title_to_split[d.title]
            self.split_to_examples[s].append(d)

    def save_examples(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for k, v in self.split_to_examples.items():
            # skip saving empty list
            if len(v) == 0:
                continue

            json_fname = self.output_json_fmt.format(split=k)
            self.logger.info(f"Saving {len(v)} data to {json_fname}")
            outfile = self.output_dir / json_fname
            with outfile.open("w", encoding="utf-8") as f:
                json.dump([vars(e) for e in v], f,
                          indent=1, ensure_ascii=False)


class ParagraphSQuAD73kSplitsBuilder(SQuAD73kSplitsBuilder):
    output_json_fmt: str = "para_73k_{split}.json"
    split_info_subdir: str = "SQuAD_73k"


class ParagraphSQuAD81kSplitsBuilder(SQuAD73kSplitsBuilder):
    output_json_fmt: str = "para_81k_{split}.json"
    split_info_subdir: str = "SQuAD_81k"
