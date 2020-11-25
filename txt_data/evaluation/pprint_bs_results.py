"""
This script is to format the beamsearch output to be easily pasted to the
Google doc file.
"""
import json
import yaml
import logging
import numpy as np


class Configs:
    input_file = "bs-results-for-qtn.json"
    selected_idxs_yaml = "selected_idxs.yaml"
    seed = 200
    question_order_yaml = "question_orders.yaml"
    output_file = "pformat_bs_results.txt"


if __name__ == "__main__":
    cfgs = Configs()
    logger = logging.getLogger()

    ############################################################
    # Read selected ids
    ############################################################
    fname = cfgs.selected_idxs_yaml
    logger.info(f"Loading {fname} ......")
    with open(fname, "r", encoding="utf-8") as f:
        selected_ids = yaml.load(f, Loader=yaml.FullLoader)

    ############################################################
    # Read beam search results
    ############################################################
    fname = cfgs.input_file
    logger.info(f"Loading {fname} ......")
    with open(fname, "r", encoding="utf-8") as f:
        bs_results = json.load(f)

    ############################################################
    # shuffle the order of the questions
    ############################################################
    rng = np.random.default_rng(cfgs.seed)
    question_orders = list()
    for i, id_ in enumerate(selected_ids):
        arr = ["hlsqg_casual", "u-PMLM_casual", "u-PMLM_random"]
        order = rng.shuffle(arr)
        question_orders.append(
            {
                "eval_set": i,
                "id": id_,
                "question_type_order": {j: t for j, t in enumerate(arr)}
            }
        )

    with open(cfgs.question_order_yaml, "w", encoding="utf-8") as f:
        yaml.dump(question_orders, f)
    ############################################################
    # prepare output file
    ############################################################
    with open(cfgs.output_file, "w", encoding="utf-8") as f:
        for i, id_ in enumerate(selected_ids):
            question_ord = question_orders[i]["question_type_order"]
            assert id_ == question_orders[i]["id"]
            result = bs_results[id_]
            context = result["fact"]["context"]

            # ----------------------------------------------------------------
            # print results
            print(f"Evaluation set {i+1}", file=f)
            print(context, file=f)
            print("", file=f)
            for i in range(len(question_ord)):
                q_txt = result[question_ord[i]]["hypothesis"]
                print(f"Question {i+1}: {q_txt}", file=f)
                print("", file=f)
