"""
Make question json file
This sript works on:
    1. Check the bleu score among the original questions and the patched questions
    2. Simple parsing

Notes
--------
Please change the squad in config
"""
import os
import sys
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from itertools import zip_longest
from nltk.translate.bleu_score import SmoothingFunction
from os.path import expanduser
from stanza.server import CoreNLPClient
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List
from nltk.tree import ParentedTree
from tqdm import tqdm


class InterroPhraseExt:
    """
    Example
    --------
    >>> s = (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBP are) (NP (PRP you))) (. ?)))
    >>> ext = InterroPhraseExt()
    >>> ext(s)
    ["How"]

    See also
    --------
    Tags:
    http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
    """

    INTERRO_TAGS = ["WDT", "WP", "WP$", "WRB"]

    # Inverted yes/no question
    SQ_TAGS = ["VBZ", "VB", "VBP", "VP", "VBD", "MD"]

    def __init__(self):
        # use hashmap to speed up
        self.interro_tags = {k: True for k in self.INTERRO_TAGS}
        self.sq_tags = {k: True for k in self.SQ_TAGS}


    def is_wh_interrogative(self, tree):
        if tree.label() in self.interro_tags:
            return True
        return False

    def is_inverse_question(self, tree):
        parent = tree.parent()
        if parent and parent.label() == "SQ" and tree.label() in self.sq_tags:
            return True
        return False

    def get_interro_word_from_close_qn(self, tree):
        """
        Get interrogative word from a close question
        """
        # check the leftmost subtree of height 2
        for t in tree.subtrees(lambda t: t.height() == 2):
            if t.label() in self.sq_tags:
                return t.leaves()
            else:
                return None

    def forward(self, parse_tree_exp: str) -> List[str]:
        """
        Extract interrogative pharses from the parsed symbolic expressions

        Parameters
        ----------
        parse_tree_exp : str
            The symbolic expressions of a constituency parse tree.
            E.g.: (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBP are) (NP (PRP you))) (. ?)))"

        Returns
        -------
        List[str]
            A list of interrogative pharses
        """
        # build tree
        tree = ParentedTree.fromstring(parse_tree_exp)

        # Wh-
        for t in tree.subtrees(filter=self.is_wh_interrogative):
            ret = t.parent().leaves()
            if ret:
                return ret

        # inverse question
        for t in tree.subtrees(filter=self.is_inverse_question):
            ret = t.leaves()
            if ret:
                return ret

        # Other missing questions
        ret = self.get_interro_word_from_close_qn(tree)
        if ret:
            return ret

        return []

    def __call__(self, parse_tree_exp: str) -> List[str]:
        return self.forward(parse_tree_exp)


class Configs:
    """
    CONFIGURATIONS
    """
    squad_dir = "../squad"
    patch_txt = "./txt/patch_questions.txt"
    blocklist_txt = "./txt/blocklist.txt"
    patch_json = "./json/patch_questions.json"
    blocklist_json = "./json/blocklist.json"
    CORENLP_HOME = "~/stanford-corenlp-4.1.0"
    memory = "4G"
    cpu_count = 4

    def __init__(self):
        self.squad_dir = Path(self.squad_dir)
        self.squad_train = self.squad_dir / "train-v1.1.json"
        self.squad_dev = self.squad_dir / "dev-v1.1.json"

        if not os.path.exists(self.patch_txt):
            ValueError("The pathed question data not exists")
        # expand ~
        self.CORENLP_HOME = expanduser(self.CORENLP_HOME)
        # add to env
        os.environ["CORENLP_HOME"] = self.CORENLP_HOME

        # limit cpu count
        self.cpu_count = min(mp.cpu_count(), self.cpu_count)

        # pathlib-ise the output filepaths
        self.patch_json = Path(self.patch_json)
        self.blocklist_json = Path(self.blocklist_json)

        # ensure output dir
        self.patch_json.parent.mkdir(parents=True, exist_ok=True)
        self.blocklist_json.parent.mkdir(parents=True, exist_ok=True)


def trim_parse_tree(s: str) -> str:
    # remove EOL
    s = s.replace("\n", " ")

    # remove extra spaces
    for _ in range(20):
        new = s.replace("  ", " ")
        if new == s:
            break
        s = new
    return s


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    """

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_id_questions(squad_data: list) -> Dict[str, str]:
    ret = dict()

    for topic in squad_data:
        for paragraph in topic["paragraphs"]:
            for qas in paragraph["qas"]:
                id_ = qas["id"]
                question_text = qas["question"]
                ret[id_] = question_text

    return ret


if __name__ == "__main__":
    # configs
    cfgs = Configs()

    # -----------------------------------
    # Load question data
    # -----------------------------------
    # load train json
    with open(cfgs.squad_train, "r") as f:
        data = json.load(f)["data"]
    train_q = load_id_questions(data)

    # load dev json
    with open(cfgs.squad_dev, "r") as f:
        data = json.load(f)["data"]
    dev_q = load_id_questions(data)
    id_to_question = {**train_q, **dev_q}

    # -----------------------------------
    # Load patched data
    # -----------------------------------
    with open(cfgs.patch_txt, "r", encoding="utf-8") as f:
        line_cache = []
        for i, line in enumerate(f):
            if i == 0:
                assert "START PATCHED QUESTIONS" in line
                continue
            if "END PATCHED QUESTIONS" in line:
                break
            line_cache.append(line)

        lines = [l.strip() for l in line_cache if l != "\n"]  # noqa: E741

    output_list = []
    smoothie = SmoothingFunction().method4
    for id_, q in grouper(lines, 2):
        ref = nltk.word_tokenize(id_to_question[id_])
        ref = [r.lower() for r in ref]
        hyp = nltk.word_tokenize(q)
        hyp = [h.lower() for h in hyp]
        bleu = sentence_bleu([ref], hyp, smoothing_function=smoothie)
        output_list.append(
            {
                "id": id_,
                "original": id_to_question[id_],
                "question": q,
                "bleu": bleu
            }
        )

    # count # questions
    n_questions: int = len(output_list)
    print(f"Collected {n_questions} questions")
    # ------------------------------------------------------------------------
    # Start parsing with corenlp

    # Interrogative phrase extractor
    ext = InterroPhraseExt()

    no_wh_questions = dict()
    with CoreNLPClient(
            annotators=['tokenize', 'parse'], threads=cfgs.cpu_count,
            memory='4G',
            endpoint='http://localhost:5501', be_quiet=True, timeout=30000) as client:

        for d in tqdm(output_list):
            question = d["question"]
            document = client.annotate(question, output_format="json")
            n_sent = len(document["sentences"])
            d["wh-phrases"] = []
            d["parse_trees"] = []
            for i in range(n_sent):

                # parse tree
                parse_tree = document['sentences'][i]['parse']
                parse_tree = trim_parse_tree(parse_tree)
                d["parse_trees"].append(parse_tree)

                # Interrogative pharse extractions
                wh_phrases = ext(parse_tree)
                d["wh-phrases"].extend(wh_phrases)

            if not d["wh-phrases"]:
                id_ = d["id"]
                no_wh_questions[id_] = question

    if no_wh_questions:
        print("--------------- WARNING ---------------")
        print("The following questions do not have wh-phrases: ")
        for k, v in no_wh_questions.items():
            print(k, v)
        print("--------------- WARNING ---------------")

    with open(cfgs.patch_json, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=1, ensure_ascii=False)
    print(f"Saved patched questions into {cfgs.patch_json}")

    # --------------------------------------------------------
    # Blocklist
    # --------------------------------------------------------
    blocklists = list()
    with open(cfgs.blocklist_txt, "r") as f:
        for line in f:
            id_, q_txt = line.strip().split(" ", maxsplit=1)
            org_q = id_to_question[id_]
            assert org_q.strip() == q_txt
            blocklists.append({
                "id": id_,
                "question": q_txt
            })

    with open(cfgs.blocklist_json, "w", encoding="utf-8") as f:
        json.dump(blocklists, f, indent=1, ensure_ascii=False)
    print(f"Saved blocklist into {cfgs.blocklist_json}")
