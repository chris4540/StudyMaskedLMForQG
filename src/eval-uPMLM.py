"""
                                Script Documentation

This script is to generate questions specific for the model `u-PMLM`


This script evaluate the following experiments:

2. u-PMLM + sequential decoding + sample question token length
3. u-PMLM + random decoding + sample question token length
4. u-PMLM + sequential decoding + given true question token length
5. u-PMLM + random decoding + given true question token length
"""
import time
import torch
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from nlgeval import NLGEval
import utils
from utils.logging import logging
from utils.logging import set_logfile_output
from utils.data.datasets.wrapper import uPMLMCondTextGenEvalDatasetWrapper
from utils.hf_argparser import HfArgumentParser
from utils.eval.arguments import BaseEvalScriptArguments
from utils.eval.decoding import CausalMLMBSCondTokenDecoder
from utils.eval.decoding import uPMLMBSCondTokenDecoder
from utils.eval.decoding import BaseBeamSearchTokenDecoder
from utils.eval.factory import BaseEvaluationFactory
from utils.eval import clean_hypothesis

LOG_FILE_NAME = "u-PMLM-eval.log"
# OUTPUT_FILE_NAME = "u-PMLM-eval-results.json"


@dataclass
class ScriptArguments(BaseEvalScriptArguments):

    random_gen_order: bool = field(
        default=False,
        metadata={
            "help": "Generate question token in random order."
        }
    )

    sample_tok_len: bool = field(
        default=False,
        metadata={
            "help": "Sample the length in token unit from a Poisson distribution"
        }
    )


class EvaluationFactory(BaseEvaluationFactory):
    """
    Simple factory to give out a correct dataset and decoder according the model
    """

    sample_params = {"poisson": {"lambda": 12.22, "min": 1}}
    num_workers: int = 2
    _device: str = "cuda"

    def __init__(self, configs: ScriptArguments):
        super().__init__(configs)
        self.random_gen_order = configs.random_gen_order
        self.sample_tok_len = configs.sample_tok_len

    def create_dataset(self):
        ds = uPMLMCondTextGenEvalDatasetWrapper(
            self.text_dataset, self.tokenizer,
            sample_decode_length=self.sample_tok_len,
            sample_params=self.sample_params)
        return ds

    def create_decoder(self) -> BaseBeamSearchTokenDecoder:
        ret: BaseBeamSearchTokenDecoder
        if self.random_gen_order:
            ret = uPMLMBSCondTokenDecoder(
                self.model, self.tokenizer,
                decode_length_known=True,
                no_repeat_ngram_size=2,
                num_return_sequences=1)
        else:  # casual / left-to-right
            ret = CausalMLMBSCondTokenDecoder(
                self.model, self.tokenizer,
                decode_length_known=True,
                no_repeat_ngram_size=2,
                num_return_sequences=1)
        return ret

    def create_task_name(self) -> str:
        if self.random_gen_order:
            _gen_order = "rand"
        else:
            _gen_order = "seq"

        if self.sample_tok_len:
            _tok_len = "smpl_len"
        else:
            _tok_len = "true_len"

        ret = f"u-PMLM-{_gen_order}-{_tok_len}"

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

    # generation results cache
    results = []

    stime = time.time()
    for inputs in tqdm(dataloader, desc=task_name):
        question_tok_ids = inputs["question"]
        question_lens = inputs["question_len"]

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
            q_len = int(question_lens[i])
            ref_question = decoder.decode(question_tok_ids[i, :])
            hyp_question = decoder.decode(decoded[i][0])
            hyp_question = clean_hypothesis(hyp_question)
            score_ = scores[i][0]
            rec = {
                "id": id_,
                "question_len": q_len,
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
