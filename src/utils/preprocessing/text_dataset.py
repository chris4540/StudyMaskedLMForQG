"""
Module to load data for preprocessing
"""
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from pprint import pformat
from typing import List
from typing import Iterator
from collections import Counter
from collections import defaultdict


@dataclass
class BaseSquadExample:
    id: str
    question: str
    context: str


@dataclass
class SquadExample(BaseSquadExample):
    """
    A data class describle one example in the SQuAD dataset.

    It would be more clear to understand the structure of the data
    """
    title: str
    is_impossible: bool
    answer_text: str
    answer_start: int

    def __repr__(self):
        return pformat(vars(self))


class SQuADTextDataset:
    """
    Stanford Question Answering Dataset

    This class is only for pre-processing only.
    No language processing like tokenization, wh-phrases extractions will be
    performed.

    Purpose
    --------
    - load the data
    - assure that each example contains certain fields
    - save down as a triplet json
    - load from a triplet json
    - select answers

    Reference
    ---------
    transformers.data.processors.squad.SquadV1Processor
    """

    train_file: str
    dev_file: str
    version: str
    example: List[SquadExample]
    allowed_splits: List[str] = ["train", "dev"]

    def __init__(self, data_dir: str, split: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = Path(data_dir)
        self.logger.info(f"Data folder: {str(self.data_dir)}")

        if split == "train":
            self.json_fname = self.train_file
        elif split == "dev":
            self.json_fname = self.dev_file
        else:
            msg = "The split should be either {0} or {1}".format(
                *self.allowed_splits)
            e = ValueError(msg)
            self.logger.critical(e)
            raise e

        # log the split
        self.logger.info(f"The split of the dataset: {split}")

        self.json_file = self.data_dir / self.json_fname
        self.logger.info(f"Reading the source data file: {self.json_file}")
        with self.json_file.open("r", encoding="utf-8") as f:
            input_data = json.load(f)["data"]

        self.examples = self._create_examples(input_data)
        self.logger.info(f"Loaded {len(self.examples)} examples")

    def __getitem__(self, key: int) -> SquadExample:
        return self.examples[key]

    def __setitem__(self, key: int, value: SquadExample):
        if not isinstance(value, SquadExample):
            raise ValueError("You must give an instance of SquadExample!")
        # set it
        self.examples[key] = value

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[SquadExample]:
        for example in self.examples:
            yield example

    def filter(self, filter_fun):
        self.examples = [
            e for e in self.examples if not filter_fun(e)
        ]

    @classmethod
    def from_json(cls, file):
        """
        An alternative constructor from a dumped file
        """
        # create a new instance.
        obj = cls.__new__(cls)
        # initialization empty object
        super(SQuADTextDataset, obj).__init__()  # Don't forget to call any polymorphic base class initializers
        #
        obj.logger = logging.getLogger(obj.__class__.__name__)
        obj.data_dir = None
        obj.json_fname = file
        # Load the json file.
        with open(file, "r", encoding="utf-8") as f:
            examples = json.load(f)
        obj.examples = [SquadExample(**e) for e in examples]
        return obj

    def to_json(self, file):
        """
        Dump examples into a json file
        """
        examples = [vars(e) for e in self.examples]
        with open(file, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=1)

    @staticmethod
    def _select_answers(answers: list):
        """
        We select answers using the following rules:
        1. voting
        2. the shortest one.
        """
        if len(answers) == 1:
            return answers[0]

        # ------------------------------------
        # Vote for the popular answer
        # ------------------------------------
        start_pos: dict = defaultdict(list)
        votes: Counter = Counter()
        for ans_dict in answers:
            answer_text = ans_dict["text"]
            ans_char_start_pos = ans_dict["answer_start"]
            start_pos[answer_text].append(ans_char_start_pos)
            votes[answer_text] += 1

        # --------------------------------------
        # if we have agreement (i.e. # of votes != 1)
        # --------------------------------------
        ans, n_vote = votes.most_common(1)[0]
        if n_vote != 1:
            return {
                "text": ans,
                "answer_start": start_pos[ans][0]
            }

        # --------------------------------------
        # if equal votes, select the shortest one
        # --------------------------------------
        min_len = 9999
        idx = -1
        for i, ans_dict in enumerate(answers):
            len_ = len(ans_dict["text"])
            if len_ > min_len:
                idx = i
                min_len = len_
        ret = {
            "text": answers[idx]["text"],
            "answer_start": answers[idx]["answer_start"]
        }
        return ret

    def _create_examples(self, input_data):
        examples = []
        for entry in input_data:
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        answers = qa["answers"]
                        answer = self._select_answers(answers)

                    example = SquadExample(
                        id=qas_id,
                        question=question_text.strip(),
                        context=context_text,
                        answer_text=answer["text"].strip(),
                        answer_start=answer["answer_start"],
                        title=title,
                        is_impossible=is_impossible
                    )

                    examples.append(example)

        return examples


class SQuADV1TextDataset(SQuADTextDataset):
    train_file: str = "train-v1.1.json"
    dev_file: str = "dev-v1.1.json"
    version = "1.1"
