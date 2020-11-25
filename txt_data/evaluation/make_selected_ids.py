import json
import logging
import numpy as np
import yaml


class Configs:
    input_ = "../preprocessed/para_73k_test.json"
    output = "selected_idxs.yaml"
    num_samples = 30
    random_seed = 662


if __name__ == "__main__":

    cfgs = Configs()

    logger = logging.getLogger()
    logger.setLevel("INFO")

    # read the test file
    with open(cfgs.input_, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # extract test data ids
    ids_ = [d["id"] for d in test_data]

    logging.info(f"# of ids in test data = {len(ids_)}")
    # --------------------------------------------------
    rnd_state = np.random.RandomState(cfgs.random_seed)
    smpl_idx = rnd_state.choice(ids_, size=cfgs.num_samples).tolist()


    # save down
    with open(cfgs.output, "w") as f:
        f.write("# This file has selected set of questions for human evaluations")
        f.write("\n")
        yaml.dump(smpl_idx, f)
