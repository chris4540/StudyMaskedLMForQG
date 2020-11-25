import numpy as np
from torch.utils.data.dataset import Dataset
from .squad import QnGenHLCtxtDataset
from . import pad_array
from utils.logging import logging


def _cast_to_long(arr):
    """
    Cast the array to np.int64
    Notes
    -----
    https://pytorch.org/docs/stable/tensors.html
    """
    # assert np.issubdtype(arr.dtype, np.int)
    ret = arr.astype(np.int64, casting='safe')
    return ret


class BaseDatasetWrapper(Dataset):

    def __init__(self, text_dataset: QnGenHLCtxtDataset, tokenizer):
        self.train = text_dataset.train
        self._ds = text_dataset
        self.tokenizer = tokenizer
        # lenght
        self._length = len(self._ds)

        # max_seq_length
        self.max_seq_length = self._ds.max_seq_length
        # max_query_length
        self.max_query_length = self._ds.max_query_length

    def __len__(self) -> int:
        return self._length


class CausalCondTextGenEvalDatasetWrapper(BaseDatasetWrapper):
    """
    A dataset wrapper for causal (left-to-right) conditional text generation evaluation

    See also
    --------
    utils.eval.decoding.CondCausalMLMTokenDecoder
    """

    def __getitem__(self, idx):
        data = self._ds[idx]

        # Notes: Since we cannot hints the length in the input for
        # causal conditional text generation, we estimate the longest possible
        # length and pad them up
        arr_len = len(data["input_ids"])
        q_start_pos = data["question_start_pos"]
        arr_maxlen = q_start_pos + self.max_query_length + 1  # last plus-one for [SEP]
        assert arr_len <= arr_maxlen
        # ----------------------
        # attention_mask
        # ----------------------
        fill_val = 0   # attention = 0 means no attention
        arr = np.array(data["attention_mask"])
        arr[q_start_pos:] = fill_val
        arr = pad_array(arr, arr_maxlen, fill_val, padding_side="right")
        attention_mask = _cast_to_long(arr)

        # ----------------------
        # token_type_ids
        # ----------------------
        fill_val = 1   # segment b, quesiton
        arr = np.array(data["token_type_ids"])
        arr = pad_array(arr, arr_maxlen, fill_val, padding_side="right")
        token_type_ids = _cast_to_long(arr)

        # ----------------------
        # input_ids
        # ----------------------
        # We pad the array full of masked tokens
        mask_token_id = self.tokenizer.mask_token_id
        arr = np.array(data["input_ids"])
        arr[q_start_pos:] = mask_token_id
        arr = pad_array(arr, arr_maxlen, mask_token_id, padding_side="right")
        input_ids = _cast_to_long(arr)

        # ----------------------
        # question tokens
        # ----------------------
        arr = np.array(data["input_ids"])[q_start_pos:]
        question_toks = arr

        # --------------------
        # id
        # --------------------
        id_ = str(data["id"])

        ret = {
            "id": id_,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "question_start_pos": int(q_start_pos),
            "question": question_toks
        }
        return ret


class uPMLMCondTextGenEvalDatasetWrapper(BaseDatasetWrapper):

    mean_decode_length: int

    def __init__(self,
                 text_dataset: QnGenHLCtxtDataset, tokenizer,
                 sample_decode_length=False,
                 sample_params={"poisson": {"lambda": 12.22, "min": 1}},
                 min_decode_length=3):

        self.sample_decode_length = sample_decode_length
        self.min_decode_length = min_decode_length
        self.sample_params = sample_params
        assert len(self.sample_params) == 1
        super().__init__(text_dataset, tokenizer)
        # ----------------------------------------------------
        # logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("-------- u-PMLM Evaluation Dataset Attributes --------")
        info_attrs = ["sample_decode_length", "min_decode_length", "sample_params"]
        for k in info_attrs:
            v = getattr(self, k)
            logger.info(f"{k:20s} : {str(v)}")
        logger.info("-------- u-PMLM Evaluation Dataset Attributes --------")

    def __getitem__(self, idx):
        data = self._ds[idx]

        arr_length = len(data["input_ids"])
        # ----------------------
        # question_start_pos
        # ----------------------
        q_start_pos = data["question_start_pos"]

        # --------------------
        # num_question_toks
        # --------------------
        if self.sample_decode_length:
            num_q_toks = self._sample_num_question_toks()
        else:
            num_q_toks = arr_length - q_start_pos - 1
        arr_maxlen = q_start_pos + num_q_toks + 1

        # ----------------------
        # attention_mask
        # ----------------------
        arr = np.array(data["attention_mask"])
        arr = pad_array(arr, arr_maxlen, 1, padding_side="right")
        arr = arr[:arr_maxlen]
        attention_mask = _cast_to_long(arr)

        # ----------------------
        # token_type_ids
        # ----------------------
        arr = np.array(data["token_type_ids"])
        arr = pad_array(arr, arr_maxlen, 1, padding_side="right")
        arr = arr[:arr_maxlen]
        arr = pad_array(arr, arr_maxlen, 1, padding_side="right")
        token_type_ids = _cast_to_long(arr)

        # ----------------------
        # input_ids
        # ----------------------
        mask_token_id = self.tokenizer.mask_token_id
        sep_token_id = self.tokenizer.sep_token_id
        arr = np.array(data["input_ids"])
        arr = pad_array(arr, arr_maxlen, mask_token_id, padding_side="right")
        arr = arr[:arr_maxlen]
        arr[q_start_pos:-1] = mask_token_id
        arr[-1] = sep_token_id
        input_ids = _cast_to_long(arr)

        # ----------------------
        # question tokens
        # ----------------------
        arr = np.array(data["input_ids"])[q_start_pos:]
        question_toks = arr

        question_len = num_q_toks  # the last [SEP] should not be count

        # --------------------
        # id
        # --------------------
        id_ = str(data["id"])

        ret = {
            "id": id_,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "question_start_pos": int(q_start_pos),
            "question_len": question_len,
            "question": question_toks
        }
        return ret

    def _sample_num_question_toks(self):
        sample_params = self.sample_params
        min_ret = self.min_decode_length
        dist_name = list(sample_params.keys())[0]
        assert dist_name in ["poisson", "normal", "constant"]
        if dist_name == "poisson":
            params = sample_params["poisson"]
            min_ = params["min"]
            lam_ = params["lambda"]
            shifted_lam = lam_ - min_
            ret = np.random.poisson(lam=shifted_lam) + min_
        elif dist_name == "normal":
            params = sample_params["normal"]
            mean = params["mean"]
            if "stddiv" in params:
                stddiv = params["stddiv"]
            else:
                var = params["var"]
                stddiv = np.sqrt(var)
            ret = np.random.normal(loc=mean, scale=stddiv)
            ret = np.ceil(ret)
        elif dist_name == "constant":
            params = sample_params["constant"]
            ret = params
        else:
            raise ValueError("sample_params must have key either `poisson` or `normal`")

        ret = max(ret, min_ret)
        ret = int(ret)
        return ret


class UniformMLMDatasetWrapper(Dataset):
    """
    Uniform masked language model dataset wrapper
    """

    def __init__(self, text_dataset: QnGenHLCtxtDataset, tokenizer):
        self.train = text_dataset.train
        self._ds = text_dataset
        self.tokenizer = tokenizer

        # lenght
        self._length = len(self._ds)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        inputs = self._ds[idx]
        ret = self._mask_question_tokens(inputs)
        return ret

    def _mask_question_tokens(self, inputs):
        """
        For training Modeling Probabilistically Masked Language Model
        specify on Question generation with uniform masking ratio prior.


        See also
        ---------
        transformers.data.data_collator.DataCollatorForLanguageModeling
        """
        input_ids = inputs["input_ids"]
        # sample masking ratio for this input
        mask_ratio = np.random.uniform(0, 1)
        # ------------------------------------------------
        q_start, q_end = np.where(input_ids == self.tokenizer.sep_token_id)[0]
        # go to the next token from [SEP], the token to seperate sentence A and B
        q_start += 1
        q_end += 1  # Ask to also predict the last token [SEP]
        question_tokens = inputs["input_ids"][q_start:q_end]

        # create masking for question part
        question_masking = np.random.choice(
            [True, False],
            size=len(question_tokens),
            p=[mask_ratio, 1-mask_ratio])

        # force we at least has one prediction for training.
        if not np.any(question_masking):
            mask_pos = np.random.choice(len(question_masking), size=1)
            question_masking[mask_pos] = True
        # assert np.any(question_masking)

        # create masking for input_ids
        masking = np.full(input_ids.shape, fill_value=False)
        masking[q_start:q_end] = question_masking

        # masking
        masked_input_ids = np.array(inputs["input_ids"])  # clone a new one
        masked_input_ids[masking] = self.tokenizer.mask_token_id
        labels = np.array(input_ids)  # clone a new one as labels
        labels[~masking] = -100  # We only compute loss on masked tokens

        # -------------------
        # Prepare return
        # -------------------
        # prepare a clone to avoid modifying the inner dataset
        attention_mask = np.array(inputs["attention_mask"])
        token_type_ids = np.array(inputs["token_type_ids"])
        ret = {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        # cast to long
        ret = {k: _cast_to_long(v) for k, v in ret.items()}
        return ret


class CondCausalMLMDatasetWrapper(Dataset):
    """
    Conditional Causal (Left-to-Right) Masked Language Model dataset wrapper
    for training or evaluation

    This is a dataset wrapper as:
        1. The question generation dataset has only unmasked triplets
        2. The text dataset is not responsible for masking
        3. The original paper uses teacher forcing and counts them into one epoch

    Reference
    ---------
    DataCollatorForLanguageModeling
    """

    def __init__(self, text_dataset: QnGenHLCtxtDataset, tokenizer, padding=False):
        self.train = text_dataset.train
        self._ds = text_dataset
        self.tokenizer = tokenizer
        self.pad_to_max = padding
        self.max_seq_length = self._ds.max_seq_length

        # --------------------------------------------
        # collect how long this dataset is
        num_qtoks = self._ds.packed_features["num_question_toks"]
        num_targets = 0
        for n in num_qtoks:
            num_targets += n + 1  # the extra one for the last [SEP]
        length = num_targets
        # --------------------------------------------
        # build cumulative sum for bisection retrival
        cumsum_qtoks = np.cumsum(np.array(num_qtoks) + 1)

        # save down as attributes
        self._length = length
        self.num_qtoks = num_qtoks
        self.cumsum_qtoks = cumsum_qtoks

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        # alias
        arr_maxlen = self.max_seq_length
        pad_to_max = self.pad_to_max
        pad_token_id = self.tokenizer.mask_token_id
        mask_token_id = self.tokenizer.mask_token_id

        # map idx to the index in text dataset
        txtds_idx = np.searchsorted(self.cumsum_qtoks, idx, side="right")
        data = self._ds[txtds_idx]

        # check the masking position
        question_start_pos = data["question_start_pos"]

        # find the start index corresponding to this text dataset item
        # Notes: txtds_idx - 1 cannot be negative
        if txtds_idx > 0:
            start_idx = self.cumsum_qtoks[txtds_idx-1]
        else:
            start_idx = 0

        mask_pos = question_start_pos + idx - start_idx

        # ----------------------
        # attention_mask
        # ----------------------
        arr = np.array(data["attention_mask"])
        arr[mask_pos:] = 0
        if pad_to_max:
            arr = pad_array(arr, arr_maxlen, 0, padding_side="right")
        attention_mask = arr

        # ----------------------
        # token_type_ids
        # ----------------------
        arr = np.array(data["token_type_ids"])
        arr[mask_pos:] = 0
        if pad_to_max:
            arr = pad_array(arr, arr_maxlen, 0, padding_side="right")
        token_type_ids = arr

        # -----------------
        # input_ids & labels
        # -----------------
        input_ids = np.array(data["input_ids"])
        labels = np.full_like(input_ids, -100)  # ignore by cross-entropy
        labels[mask_pos] = input_ids[mask_pos]
        input_ids[mask_pos:] = pad_token_id
        input_ids[mask_pos] = mask_token_id
        if pad_to_max:
            labels = pad_array(labels, arr_maxlen, -100, padding_side="right")
            input_ids = pad_array(input_ids, arr_maxlen, pad_token_id, padding_side="right")

        # build return
        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        ret = {k: _cast_to_long(v) for k, v in ret.items()}
        return ret
