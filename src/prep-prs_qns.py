"""
Create parsed_dep_questions.json

Prerequisite
--------------
Two files from patching:
    patched_train-v1.1.json
    patched_dev-v1.1.json

parsed_dep_questions.json:
----------------------------
[
 {
  "id": "5733be284776f41900661182",
  "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
  "parse_trees": [
   "(ROOT (SBARQ (WHPP (TO To) (WHNP (WP whom))) (SQ (VBD did) (NP (DT the) (NNP Virgin) (NNP Mary)) (ADVP (RB allegedly)) (VP (VBP appear) (PP (IN in) (NP (NP (CD 1858)) (PP (IN in) (NP (NNP Lourdes) (NNP France))))))) (. ?)))"
  ]
 }, ...
]

TODO
-------
Write this script better
"""
import json
from utils.preprocessing.text_dataset import SQuADV1TextDataset
from stanza.server import CoreNLPClient
from tqdm import tqdm
import os

os.environ["CORENLP_HOME"] = "/home/chrislin/stanford-corenlp"


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


def get_questions(json_):
    ret = []
    ds = SQuADV1TextDataset.from_json(json_)
    for d in ds:
        ret.append({
            "id": d.id,
            "question": d.question
        })
    return ret


if __name__ == "__main__":
    # collect questions
    q_from_train = get_questions(
        "../txt_data/preprocessed/patched_train-v1.1.json")
    q_from_dev = get_questions(
        "../txt_data/preprocessed/patched_dev-v1.1.json")

    questions = [*q_from_train, *q_from_dev]

    with CoreNLPClient(
            annotators=['tokenize', 'parse'], threads=4,
            memory='12G',
            endpoint='http://localhost:5501', be_quiet=True, timeout=30000) as client:

        for d in tqdm(questions):
            question = d["question"]
            document = client.annotate(question, output_format="json")
            n_sent = len(document["sentences"])
            d["parse_trees"] = []
            for i in range(n_sent):
                # parse tree
                parse_tree = document['sentences'][i]['parse']
                parse_tree = trim_parse_tree(parse_tree)
                d["parse_trees"].append(parse_tree)

    with open("parsed_dep_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=1, ensure_ascii=False)
