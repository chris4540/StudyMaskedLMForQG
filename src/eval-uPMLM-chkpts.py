"""
Use it for evaluate checkpoints
"""
from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from argparse import Namespace
from utils import DataArguments
from utils.train import TrainingArguments
from utils.hf_argparser import HfArgumentParser
from utils.hf_argparser import save_cfg_dict
from utils.logging import logging
from utils.logging import set_logfile_output
from utils import is_format_string
from models.bert_qgen import BertForMaskedLM
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.data.datasets.wrapper import UniformMLMDatasetWrapper
from utils.data.datasets.wrapper import uPMLMCondTextGenEvalDatasetWrapper
from utils.data.datasets.data_collator import DataCollatorForPadding
from utils.eval.decoding import CausalMLMCondTokenDecoder
from utils.train.trainer import Trainer
from utils import get_chkpts_from


@dataclass
class ScriptArguments:
    """
    This script argument
    """
    cfg: str = field(
        metadata={
            "help": "The experiment config file. e.g. <exp_name>.yaml"
        }
    )


if __name__ == "__main__":
    # ---------------------------------------
    # Script configuration
    # ---------------------------------------
    # Pick up the arguments
    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()

    # set the config file
    cfgfile = args.cfg
    logger = logging.getLogger()
    logger.info(f"Using the config file: {cfgfile}")

    # read config file
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    cfgs_from_file = parser.parse_yaml_file(cfgfile)

    # ----------------------------------------
    # Make configs a simple namespace
    # ----------------------------------------
    cfg_dict = dict()
    for c in cfgs_from_file:
        cfg_dict.update(dataclasses.asdict(c))

    # print(cfg_dict)
    # parse the format string in the config file
    for k, v in cfg_dict.items():
        if is_format_string(v):
            cfg_dict[k] = v.format(**cfg_dict)

    # build config dict
    cfg_dict.update(dataclasses.asdict(args))

    # build a namespace
    cfg = Namespace(**cfg_dict)

    # -----------------
    # logging
    # -----------------
    logger.setLevel(cfg.loglevel.upper())
    logging_dir = Path(cfg.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logfile = logging_dir / "train.log"
    set_logfile_output(logfile)

    # save arguments summary
    save_cfg_dict(cfg_dict)

    # rebuild train args
    training_args = TrainingArguments(**cfg_dict)

    # -----------------
    # tokenizer
    # -----------------
    tokenizer = BertTokenizerWithHLAns.from_pretrained(cfg.model_name)

    # -------------
    # Dataset
    # -------------
    txtds_cache_dir = Path(cfg.txtds_cache_dir)
    # -----------------
    # train dataset
    # -----------------
    train_txtds = QnGenHLCtxtDataset.from_cache(txtds_cache_dir / "train_ds")
    train_ds = UniformMLMDatasetWrapper(train_txtds, tokenizer)
    # -----------------
    # validation dataset
    # -----------------
    dev_txtds = QnGenHLCtxtDataset.from_cache(txtds_cache_dir / "dev_ds")
    dev_ds = UniformMLMDatasetWrapper(dev_txtds, tokenizer)
    dev_gen_ds = uPMLMCondTextGenEvalDatasetWrapper(dev_txtds, tokenizer, sample_decode_length=False)
    # -----------------
    # test dataset
    # -----------------
    test_txtds = QnGenHLCtxtDataset.from_cache(txtds_cache_dir / "test_ds")
    test_ds = UniformMLMDatasetWrapper(test_txtds, tokenizer)
    test_gen_ds = uPMLMCondTextGenEvalDatasetWrapper(test_txtds, tokenizer, sample_decode_length=True)

    # data collator
    pad_collator = DataCollatorForPadding(tokenizer)

    # text generation decoder
    max_decode_len = dev_gen_ds.max_query_length
    lm_gen_decoder = CausalMLMCondTokenDecoder(
        tokenizer=tokenizer, max_decode_len=max_decode_len,
        num_return_sequences=1, decode_length_known=True)

    # do checkpoint evaluation
    for steps, model_path in get_chkpts_from(cfg.output_dir, reverse=False):
        model = BertForMaskedLM.from_pretrained(model_path)
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=pad_collator,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            eval_gen_dataset=dev_gen_ds,
            eval_gen_decoder=lm_gen_decoder,
            prediction_loss_only=False,
        )
        trainer.global_step = steps
        logger.info(f"eval-uPMLM-chkpts: Perform evaluation at checkpoint-{steps}")
        trainer.evaluate()
        # do prediction
        trainer.text_generation(test_gen_ds)
        trainer.predict(test_ds)
