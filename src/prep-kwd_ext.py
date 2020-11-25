"""
Prerequisite
--------------
Two files from patching:
    para_73k_train.json

Output
-------
    para_73k_train.json

Aims
----------
- keyword extraction
- interrogative phrases extraction
"""
# from models.tokenizer import BertTokenizerWithHLAns
from utils.preprocessing.text_dataset import SQuADV1TextDataset
import json
from collections import Counter
from collections import OrderedDict
from utils.preprocessing.extractor import TokensExt
from utils.preprocessing.extractor import InterroPhraseExt
from nltk.corpus import stopwords
from tqdm import tqdm
from pathlib import Path


class Config:
    ds_json = "../txt_data/preprocessed/para_73k_train.json"
    parsed_q_json = "../txt_data/preprocessed/parsed_dep_questions.json"
    output_file = "para_73k_train.json"
    work_dir = "./prep/proc_qns"


def load_parsed_dep_questions():
    cfg = Config()
    with open(cfg.parsed_q_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    ret = dict()
    for d in data:
        id_ = d.pop("id")
        ret[id_] = d
    return ret


def ext_keywords_interro_phrs(parse_trees):
    wh_phrases = set()
    keywords = set()
    for tree in parse_trees:
        toks = [x.lower() for x in toks_ext(tree)]
        # wh-words
        wh_phrase = [x.lower() for x in interro_phrase_ext(tree)]
        wh_phrases.update(wh_phrase)

        kw = [x for x in toks if x in filtered_freq]
        keywords.update(kw)
    keywords = keywords - wh_phrases
    # -----------------------------------------------------------
    ret = {
        "keywords": list(keywords),
        "wh_phrases": list(wh_phrases)
    }

    return ret


if __name__ == "__main__":
    cfg = Config()
    ds = SQuADV1TextDataset.from_json(cfg.ds_json)
    question_dep = load_parsed_dep_questions()

    term_freq: Counter = Counter()
    toks_ext = TokensExt()
    interro_phrase_ext = InterroPhraseExt()

    for d in ds:
        question = d.question
        id_ = d.id
        parse_trees = question_dep[id_]["parse_trees"]
        for tree in parse_trees:
            toks = toks_ext(tree)
            for t in toks:
                term_freq[t.lower()] += 1

    # keep only rare words as keywords
    stops = stopwords.words('english')
    stops.extend(["-lrb-", "-rrb-", "--"])
    stops = set(stops)
    filtered_freq = Counter(
        {
            w: cnt
            for w, cnt in term_freq.items()
            if cnt <= 1000 and w not in stops and len(w) > 0
        }
    )

    outs = []
    for d in tqdm(ds, desc="keyword_extraction"):
        question = d.question
        id_ = d.id
        parse_trees = question_dep[id_]["parse_trees"]
        phrases_dict = ext_keywords_interro_phrs(parse_trees)
        result = dict(vars(d))
        result.pop("is_impossible", None)
        for k, v in phrases_dict.items():
            result[k] = v
        outs.append(result)

    # save it
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    with open(work_dir / cfg.output_file, "w", encoding="utf-8") as f:
        json.dump(outs, f, indent=1, ensure_ascii=False)

    with open(work_dir / "keywd_freq.json", "w", encoding="utf-8") as f:
        json.dump(OrderedDict(filtered_freq.most_common()), f, indent=1, ensure_ascii=False)
