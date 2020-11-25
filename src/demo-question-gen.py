"""
This script is to generate
"""
import os
import torch
from pathlib import Path
from dataclasses import dataclass
from utils import save_as_json
from utils.logging import logging
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.eval import clean_hypothesis
from utils.logging import set_logfile_output
from tqdm import tqdm
from utils.eval.factory import BaseEvaluationFactory
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from utils.data.datasets.wrapper import uPMLMCondTextGenEvalDatasetWrapper
from utils.eval.decoding import CausalMLMBSCondTokenDecoder

DEMO_EXAMPLE = {
    "id": "for-demo-in-thesis",
    "context": "KTH was established in 1827 as Teknologiska Institutet (Institute of Technology), "
               "and had its roots in Mekaniska skolan (School of Mechanics) that was established in 1798 in Stockholm. "
               "But the origin of KTH dates back to the predecessor to Mekaniska skolan, the Laboratorium Mechanicum, "
               "which was established in 1697 by Swedish scientist and innovator Christopher Polhem. "
               "Laboratorium Mechanicum combined education technology, a laboratory and an exhibition space for innovations.[4] "
               "In 1877 KTH received its current name, Kungliga Tekniska högskolan (KTH Royal Institute of Technology). "
               "The King of Sweden Carl XVI Gustaf is the High Protector of KTH.",
    "title": "KTH_Royal_Institute_of_Technology",
    "answer_text": "Teknologiska Institutet (Institute of Technology)"
}

LOG_FILE_NAME = "demo-gen.log"


@dataclass
class Configs:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configurations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_name = "hlsqg"
    # model_name = "uPMLM"
    model_paths = {
        "hlsqg": "hlsqg-p73k-base-out",
        "uPMLM": "uPMLM-p73k-base-out",
    }

    tokenizer_name = "bert-base-uncased"
    batch_size = 1
    demo_text_input_path = "demo-cache.json"
    # question token length sampling params
    sample_params = {"constant": 30}
    logging_dir = "./"
    dataset_kwargs = {
        "max_seq_length": 384,
        "max_query_length": 30,
        "doc_stride": 128,
        "processes": 4,
    }


def create_complete_example(example: dict):
    ans_txt = example["answer_text"]
    context = example["context"]
    ans_start = context.find(ans_txt)
    ret = {k: v for k, v in example.items()}
    ret["is_impossible"] = False
    ret["answer_start"] = ans_start
    ret["question"] = "Not applicable"
    return ret


class DemoToolsFactory(BaseEvaluationFactory):
    """
    Simple factory to give out a correct dataset and decoder according the model
    """

    sample_params = {"poisson": {"lambda": 12.22, "min": 1}}
    num_workers: int = 2
    _device: str = "cuda"

    def __init__(self, configs):
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_name = self.configs.model_name

        # -------------------
        # tokenizer
        # -------------------
        tokenizer = BertTokenizerWithHLAns.from_pretrained(configs.tokenizer_name)
        self.tokenizer = tokenizer

        # -------------------
        # text dataset
        # -------------------
        text_dataset = QnGenHLCtxtDataset(
            input_json=configs.demo_text_input_path,
            tokenizer=tokenizer,
            **configs.dataset_kwargs
        )
        self.text_dataset = text_dataset

        # sample_params
        if self.model_name == "uPMLM":
            self.sample_params = configs.sample_params
        else:
            self.sample_params = None

        # model
        self.model_path = self.configs.model_paths[self.model_name]
        self.configs.model_path = self.model_path
        self.model = self._create_model()

    def create_dataset(self):
        """
        return a simple left-to-right generation dataset
        """
        if self.model_name == "hlsqg":
            ret = CausalCondTextGenEvalDatasetWrapper(self.text_dataset, self.tokenizer)
        else:
            ret = uPMLMCondTextGenEvalDatasetWrapper(
                self.text_dataset, self.tokenizer,
                sample_decode_length=True,
                sample_params=self.sample_params)
        return ret

    def create_decoder(self):
        if self.model_name == "hlsqg":
            decode_length_known = False
        else:
            decode_length_known = True
        ret = CausalMLMBSCondTokenDecoder(
            self.model, self.tokenizer,
            num_beams=3,
            no_repeat_ngram_size=2,
            decode_length_known=decode_length_known,
            num_return_sequences=1)
        return ret


if __name__ == "__main__":

    # tokenizer
    logger = logging.getLogger()
    configs = Configs()
    # ---------------------------------------
    # logger
    # ---------------------------------------
    logger = logging.getLogger()
    logging_dir = Path(configs.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logfile = logging_dir / LOG_FILE_NAME
    set_logfile_output(logfile)

    #########################
    # Save temporary dataset
    #########################
    # make the complete example
    example = create_complete_example(DEMO_EXAMPLE)
    # save it to json
    save_as_json(configs.demo_text_input_path, [example])
    logger.info("Saving the demo example into " + configs.demo_text_input_path)

    #
    factory = DemoToolsFactory(configs)
    # --------------------------------------------
    # Generation
    # --------------------------------------------
    # Start generation part
    model = factory.model
    dataloader = factory.create_dataloader()
    decoder = factory.create_decoder()

    for inputs in tqdm(dataloader):
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
            hyp_question = decoder.decode(decoded[i][0])
            hyp_question = clean_hypothesis(hyp_question)
            score_ = scores[i][0]
            rec = {
                "hypothesis": hyp_question,
                "score": score_,
                "answer_text": DEMO_EXAMPLE["answer_text"],
                "model_path": factory.model_path,
                "sample_params": factory.sample_params
            }
            logger.info(f"Generation result: {rec}")

    # -----------------------
    # Remove the temp file
    # -----------------------
    try:
        os.remove(configs.demo_text_input_path)  # type: ignore[arg-type]
    except IOError:
        pass
