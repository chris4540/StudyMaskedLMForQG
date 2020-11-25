"""
Script to train hlsqg causal model

Experiment 1:
    baseline
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
from utils import get_lastest_chkpt_from
from models.bert_qgen import BertForMaskedLM
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.data.datasets.wrapper import CondCausalMLMDatasetWrapper
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from utils.data.datasets.data_collator import DataCollatorForPadding
from utils.eval.decoding import CausalMLMCondTokenDecoder
from utils.train.trainer import Trainer


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

    force_eval: bool = field(
        default=False,
        metadata={
            "help": "Force to evaluate the final model only"
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

    # ------------------------------
    # model
    # ------------------------------
    model_path = get_lastest_chkpt_from(cfg.output_dir)
    if not model_path:
        logger.info(
            f"Cannot find checkpoint from {cfg.output_dir}. "
            "Using the pretrained one"
        )
        # using the pretrained one
        model = BertForMaskedLM.from_pretrained(cfg.model_name)
        # resize it
        model.resize_type_token_embeddings(3)
    else:
        model = BertForMaskedLM.from_pretrained(model_path)
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
    train_ds = CondCausalMLMDatasetWrapper(train_txtds, tokenizer)
    # -------------------
    # validation dataset
    # -------------------
    dev_txtds = QnGenHLCtxtDataset.from_cache(txtds_cache_dir / "dev_ds")
    dev_ds = CondCausalMLMDatasetWrapper(dev_txtds, tokenizer)
    dev_gen_ds = CausalCondTextGenEvalDatasetWrapper(dev_txtds, tokenizer)
    # -----------------
    # test dataset
    # -----------------
    test_txtds = QnGenHLCtxtDataset.from_cache(txtds_cache_dir / "test_ds")
    test_ds = CondCausalMLMDatasetWrapper(test_txtds, tokenizer)
    test_gen_ds = CausalCondTextGenEvalDatasetWrapper(test_txtds, tokenizer)

    # data collator
    pad_collator = DataCollatorForPadding(tokenizer)

    # text generation decoder
    max_decode_len = dev_gen_ds.max_query_length
    lm_gen_decoder = CausalMLMCondTokenDecoder(
        tokenizer=tokenizer, max_decode_len=max_decode_len,
        num_return_sequences=1, decode_length_known=False)
    # Initialize our Trainer
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

    if training_args.do_train:
        trainer.train(model_path)
        trainer.save_model()
    else:
        # --------------------
        # Check the output dir
        # --------------------
        assert training_args.output_dir
        model = BertForMaskedLM.from_pretrained(training_args.output_dir)
        trainer.model = model
        # calculate the number of step to reach the final model
        train_dataloader = trainer.get_train_dataloader()
        t_total = int(
            len(train_dataloader)
            // training_args.gradient_accumulation_steps
            * training_args.num_train_epochs)
        trainer.global_step = t_total
        logger.info(f"Perform evaluation on the trained model at step {t_total}")

    # turn off sampling
    trainer.args.sample_gen_eval = False
    trainer.evaluate()
    # do prediction
    trainer.text_generation(test_gen_ds)
    trainer.predict(test_ds)
