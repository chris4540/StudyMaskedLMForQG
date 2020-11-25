# from stanza.server import CoreNLPClient
import json
from utils.preprocessing.extract_interrogative import InterroPhraseExt


file = "../txt_data/preprocessed/parsed_dep_questions.json"

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open("patch.txt", "r", encoding="utf-8") as f:
    content = f.readlines()
ids = [x.strip() for x in content]
ids = set(ids)


ext = InterroPhraseExt()
cnt = 0
for i, d in enumerate(data):
    id_ = d["id"]
    if id_ in ids:
        continue

    tree_str = d["parsetree"]
    w = ext(tree_str)

    if not w:
        # print(i)
        # print(w)
        print(d["id"])
        # print(d["question"])
        # print(id_, d["question"])
        # if d["question"].strip().endswith("?"):
        #     print(id_, d["question"])
        #     cnt += 1
        # cnt += 1
        # print()
        # if d["question"].startswith("WH"):
        #     print(d["id"])
        #     print(d["question"])
        # print()
        # if cnt > 10:
        #     break


print("Cnt = ", cnt)
