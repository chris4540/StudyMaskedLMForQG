import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
from typing import List, Dict
from . import pad_array


class DataCollatorForPadding:
    """
    For padding use.

    Params
    ------
    [....]
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return self.collate_batch(batch)

    def _pad_arr_to(self, arr, to_length):
        ret = pad_array(arr, to_length, self.pad_token_id, padding_side="right")
        return ret


    def collate_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Pad the batch to the maximum size

        See also:
        ---------
        torch.utils.data._utils.collate.default_collate
        """
        # find the max length
        max_len = 0
        for d in batch:
            len_ = len(d["input_ids"])
            if len_ > max_len:
                max_len = len_

        # build the padded_batch
        padded_batch = []
        for d in batch:
            padded_sample = dict()
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    v = self._pad_arr_to(v, max_len)
                padded_sample[k] = v
            padded_batch.append(padded_sample)

        return default_collate(padded_batch)


# class DataCollatorForProbMaskedLangModel:
#     """
#     Data Collator for training Modeling Probabilistically Masked Language Model
#     specify on Question generation with uniform masking ratio prior.


#     See also
#     ---------
#     transformers: transformers.data.data_collator.DataCollatorForLanguageModeling
#     """

#     def __init__(self, tokenizer: PreTrainedTokenizer):
#         self.tokenizer = tokenizer

#         # TODO: think about the pad_collator
#         #       we can have a better collator
#         self.pad_collator = DataCollatorForPadding(tokenizer)

#     def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
#         return self.collate_batch(batch)

#     def mask_question_tokens(self, inputs: Dict) -> Dict:
#         """
#         Check more on:
#         mask_tokens
#         """
#         input_ids = inputs["input_ids"]
#         # sample masking ratio for this input
#         mask_ratio = np.random.uniform(0, 1.0)
#         # ------------------------------------------------
#         q_start, q_end = np.where(input_ids == self.tokenizer.sep_token_id)[0]
#         q_start += 1  # go to the next token from [SEP]
#         question_tokens = inputs["input_ids"][q_start:q_end]
#         n_question_toks = len(question_tokens)

#         # create masking for question part
#         question_masking = np.random.choice(
#             [True, False],
#             size=len(question_tokens),
#             p=[mask_ratio, 1-mask_ratio])

#         # create masking for input_ids
#         masking = np.full(input_ids.shape, fill_value=False)
#         masking[q_start:q_end] = question_masking

#         # masking
#         masked_input_ids = np.array(inputs["input_ids"])  # clone a new one
#         masked_input_ids[masking] = self.tokenizer.mask_token_id
#         labels = np.array(input_ids)  # clone a new one as labels
#         labels[~masking] = -100  # We only compute loss on masked tokens

#         # calculate the probability of the drawn masking (relative to global)
#         # n_mask_toks = np.count_nonzero(masking)
#         # binom_coeff = scipy.special.binom(n_question_toks, n_mask_toks)
#         # inv_mask_prob: the inverse of the masking probability p(M)
#         # n+1 count the number of possible ways to mask the question with length N
#         # i.e. count([0 ... N])
#         # See the paper appendix
#         # inv_mask_prob = binom_coeff * (n_mask_toks + 1)

#         # Prepare return
#         ret = dict(inputs)
#         # ret["mask_prob"] = 1.0 / inv_mask_prob
#         ret["input_ids"] = masked_input_ids
#         ret["labels"] = labels
#         return ret

#     def collate_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:

#         # mask examples one-by-one
#         masked_batch = [self.mask_question_tokens(b) for b in batch]

#         # pad it and tensorize the batch
#         ret = self.pad_collator(masked_batch)

#         # As the paddings are 0 for `input_ids` and `token_type_ids`
#         # but labels should be -100
#         labels = ret["labels"]
#         labels[labels == self.tokenizer.pad_token_id] = -100
#         ret["labels"] = labels
#         return ret


# class DataCollatorForQuestionLengthPrediction:
#     """
#     Data Collator for training Modeling Probabilistically Masked Language Model
#     specify on Question generation with uniform masking ratio prior.


#     See also
#     ---------
#     transformers: transformers.data.data_collator.DataCollatorForLanguageModeling
#     """

#     def __init__(self, tokenizer: PreTrainedTokenizer):
#         self.tokenizer = tokenizer

#         # TODO: think about the pad_collator
#         #       we can have a better collator
#         self.pad_collator = DataCollatorForPadding(tokenizer)

#     def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
#         return self.collate_batch(batch)

#     def convert_to_pred_q_len(self, inputs: Dict) -> Dict:
#         """
#         Check more on:
#         mask_tokens
#         """
#         input_ids = inputs["input_ids"]
#         # ------------------------------------------------
#         q_start, q_end = np.where(input_ids == self.tokenizer.sep_token_id)[0]
#         q_start += 1  # go to the next token from [SEP]
#         question_tokens = inputs["input_ids"][q_start:q_end]
#         n_q_tokens = len(question_tokens)
#         # ------------------------------------------------
#         # filter out quesiton
#         ret = {k: v[:q_start] for k, v in inputs.items()}

#         ret["n_q_tokens"] = n_q_tokens
#         return ret

#     def collate_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:

#         # mask examples one-by-one
#         masked_batch = [self.convert_to_pred_q_len(b) for b in batch]

#         # pad it and tensorize the batch
#         ret = self.pad_collator(masked_batch)
#         return ret


# if __name__ == "__main__":
#     pass
#     """
#     from models.tokenizer import BertTokenizerWithHLAns
#     from utils.data.datasets.uniform_masking import ContextHLAnsQuestionTextDataset
#     from transformers import set_seed
#     from utils.data.datasets.data_collator import DataCollatorForProbMaskedLangModel

#     set_seed(43)
#     file = "../txt_data/preprocessed/sentence_73k_dev.json"
#     tokenizer = BertTokenizerWithHLAns.from_pretrained("bert-base-uncased")
#     ds = ContextHLAnsQuestionTextDataset(
#         tokenizer,
#         file,
#         cache_dir="tmp_cache",
#         max_seq_length=300,
#         doc_stride=128,
#         max_query_length=64)
#     # print(ds.get_decoded_example(0))
#     # print(ds.get_decoded_example(1))
#     # -------------------------------------------
#     examples = [ds[i] for i in [0, 2, 4]]
#     # print(examples)
#     # pad_collator = DataCollatorForPadding(tokenizer)
#     # print(pad_collator(examples))
#     coll = DataCollatorForProbMaskedLangModel(tokenizer)
#     print(coll(examples))
#     """
