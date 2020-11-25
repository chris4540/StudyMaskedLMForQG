"""
                                Script Documentation

This script is to generate questions specific for the model `hlsqg`


This script evaluate the following experiments:

1. HLSQG + sequential decoding
"""
import torch
import time
import dataclasses
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from nlgeval import NLGEval
import utils
from utils.logging import logging
from utils.logging import set_logfile_output
from utils.hf_argparser import HfArgumentParser
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from utils.eval import clean_hypothesis
from utils.eval.factory import BaseEvaluationFactory
from utils.eval.arguments import BaseEvalScriptArguments
from utils.eval.decoding import CausalMLMBSCondTokenDecoder

LOG_FILE_NAME = "hlsqg-eval.log"


@dataclass
class ScriptArguments(BaseEvalScriptArguments):
    pass


class EvaluationFactory(BaseEvaluationFactory):

    def __init__(self, configs: ScriptArguments):
        super().__init__(configs)

    def create_dataset(self):
        """
        return a simple left-to-right generation dataset
        """
        ret = CausalCondTextGenEvalDatasetWrapper(self.text_dataset, self.tokenizer)
        return ret

    def create_decoder(self):
        ret = CausalMLMBSCondTokenDecoder(
            self.model, self.tokenizer,
            no_repeat_ngram_size=2,
            decode_length_known=False,
            num_return_sequences=1)
        return ret

    def create_task_name(self) -> str:
        ret = "hlsqg"
        return ret


if __name__ == "__main__":
    # ---------------------------------------
    # Script configuration
    # ---------------------------------------
    # Pick up the arguments
    parser = HfArgumentParser((ScriptArguments,))
    configs, = parser.parse_args_into_dataclasses()

    # ---------------------------------------
    # logger
    # ---------------------------------------
    logger = logging.getLogger()
    logging_dir = Path(configs.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logfile = logging_dir / LOG_FILE_NAME
    set_logfile_output(logfile)

    # ---------------------------------------
    # factory
    # ---------------------------------------
    factory = EvaluationFactory(configs=configs)

    # Start generation part
    model = factory.model
    dataloader = factory.create_dataloader()
    decoder = factory.create_decoder()
    task_name = factory.create_task_name()

    # --------------------------------------------
    # Generation
    # --------------------------------------------

    # results: generation results cache
    results = []
    stime = time.time()
    for inputs in tqdm(dataloader, desc=task_name):
        question_tok_ids = inputs["question"]

        # to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)

        # decoding
        decoded, scores = decoder(inputs)

        # check batch_size
        batch_size = question_tok_ids.shape[0]
        # loop over the batch
        for i in range(batch_size):
            id_ = inputs["id"][i]
            ref_question = decoder.decode(question_tok_ids[i, :])
            hyp_question = decoder.decode(decoded[i][0])
            hyp_question = clean_hypothesis(hyp_question)
            score_ = scores[i][0]
            rec = {
                "id": id_,
                "reference": ref_question,
                "hypothesis": hyp_question,
                "score": score_
            }
            results.append(rec)
        # break
    etime = time.time()
    generation_duration = etime - stime
    # --------------------------------------------------
    # run nlg_eval to get n-gram overlapping metrics
    n = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True)

    # build refs vs hyps
    ref_list = []
    hyp_list = []
    for rec in results:
        ref_list.append(rec["reference"])
        hyp_list.append(rec["hypothesis"])

    stime = time.time()
    metrics = n.compute_metrics(ref_list=[ref_list], hyp_list=hyp_list)
    etime = time.time()
    eval_duration = etime - stime

    output = {
        "task_name": task_name,
        "evalution_configs": dataclasses.asdict(configs),
        "generation_duration": generation_duration,
        "nlg-eval_duration": eval_duration,
        "batch_size": configs.batch_size,
        "metrics": metrics,
        "results": results,
    }

    output_fname = factory.create_output_filename()
    output_path = logging_dir / output_fname
    logging.info(f"Saving the results to {output_path}...")
    utils.save_as_json(output_path, output)
