from dataclasses import dataclass, field

@dataclass
class BaseEvalScriptArguments:
    """
    This evaluation script basic argument
    """

    model_path: str = field(
        metadata={
            "help": "The folder of the trained `BertForMaskedLM` model"
        }
    )

    logging_dir: str = field(
        metadata={
            "help": "The logging folder for evaluation."
        }
    )

    batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size of the input sequences to feed into the model"
        }
    )

    txt_ds_path: str = field(
        default="cached_txtds/test_ds",
        metadata={
            "help": "The text dataset in numpy data format"
        }
    )

    tokenizer_name: str = field(
        default="bert-base-uncased",
        metadata={
            "help": (
                "The name of the pretrained tokenizer. "
                "Will download the tokenizer data from huggingface storage."
            )
        }
    )
