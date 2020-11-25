
from .text_dataset import SQuADTextDataset
from .text_dataset import SquadExample
from typing import List
import logging
import json


class PatchedSQuADTextDataset(SQuADTextDataset):

    def __init__(self, ds: SQuADTextDataset, blocklist_file: str, patched_questions: str):
        # rename myself
        self.__class__.__name__ = f"Patched{ds.__class__.__name__}"
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Loading blocklist ids ....")
        self.blocklist_ids = set(
            [d["id"] for d in self.load_json(blocklist_file)]
        )

        self.logger.info("Loading patched questions ...")
        self.patched_questions = {
            d["id"]: d
            for d in self.load_json(patched_questions)
        }

        self.examples: List[SquadExample] = list(ds.examples)
        self.apply_blocklist()
        self.apply_patch_questions()

    def apply_blocklist(self):
        prev_n_examples = len(self.examples)
        self.filter(lambda d: d.id in self.blocklist_ids)
        n_examples = len(self.examples)
        n_removed = prev_n_examples - n_examples
        self.logger.info(f"Removed {n_removed} questions.")

    def apply_patch_questions(self):
        cnt = 0
        for i, d in enumerate(self.examples):
            if d.id in self.patched_questions:
                d.question = self.patched_questions[d.id]["question"]
                cnt += 1
        self.logger.info(f"Patched {cnt} questions.")

    @staticmethod
    def load_json(file):
        with open(file, "r", encoding="utf-8") as f:
            ret = json.load(f)
        return ret
