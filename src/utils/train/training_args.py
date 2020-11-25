# from transformers import TrainingArguments
import transformers
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields


@dataclass
class TrainingArguments(transformers.TrainingArguments):

    exp_name: str = field(
        default="unnamed-exp",
        metadata={"help": "The experiment name"}
    )
    loglevel: str = field(
        default="info",
        metadata={"help": "The experiment logging level"}
    )

    model_name: str = field(
        default="bert-base-uncased", metadata={
            "help": "the name of the model, which for loading from pretrained"
        }
    )

    # ---------------------------------
    # Text generation evaluation
    # ---------------------------------
    # do_gen_eval
    do_gen_eval: bool = field(
        default=False, metadata={
            "help": "Whether to run text generation evaluation on the dev set."
        }
    )

    # sampling
    sample_gen_eval: bool = field(
        default=True, metadata={
            "help": "Whether to sample data when performing text generation evaluation."
        }
    )
    n_sample_gen_eval: int = field(
        default=200, metadata={
            "help":
                "The number of samples when performing text generation evaluation."
        }
    )

    # per_device_geneval_batch_size
    per_device_gen_eval_batch_size: int = field(
        default=1, metadata={
            "help": "Batch size per GPU/TPU core/CPU for text generation evaluation."
        }
    )

    def __init__(self, **kwargs):
        """
        Customized constructor accepts extra unused arguments
        """
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

    @property
    def gen_eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_gen_eval_batch_size
        return per_device_batch_size * max(1, self.n_gpu)
