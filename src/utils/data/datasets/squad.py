"""

"""
from torch.utils.data.dataset import Dataset
import json
from utils.data.datasets import find_idx_of_span_in
from utils.data.datasets.doc_span import create_docspans
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from collections import defaultdict
from utils.logging import logging
from utils.logging import DummyLogger
import sys
from pathlib import Path
from typing import Union

# module constants
NUM_CPUS = mp.cpu_count()


def get_logger(logger_name) -> logging.Logger:
    """
    Make a multiprocessing friendly logger
    """
    # logger
    logger = logging.getLogger(logger_name)

    # check python version
    if sys.version_info.major == 3 and sys.version_info.minor == 6:
        logger.warning("Python 3.6 does not support picking logger. "
                       "Logging is disabled.")
        logger = DummyLogger()

    return logger


def get_hl_ans_span(context_toks, hl_token):
    """
    Return
    -------
    A tuple of start and end of the span of the highlighted answer span
    """
    hl_cnt = 0
    start = -1
    end = -1
    for i, t in enumerate(context_toks):
        if t != hl_token:
            continue

        hl_cnt += 1
        # mark as first encounter
        if hl_cnt == 1:
            start = i

        if hl_cnt >= 2:
            end = i
            break
    return start, end


def highlight_answer(input_example: dict) -> dict:
    """
    Highlighting the answer in the input example


    Args:
        input_example (dict): A single training/test example in
            question-context-answer tuple
    """
    context = input_example['context']
    answer_start = input_example['answer_start']
    answer_text = input_example['answer_text']

    hl_context = context[:answer_start] + "[HL]" + answer_text + \
        "[HL]" + context[len(answer_text)+answer_start:]

    # clone the input
    ret = dict(input_example)
    ret['highlighted_context'] = hl_context
    return ret


class CtxtQnAnsTripletEx:
    """
    (Context, Question, Answer) example
    """

    optional_attr = ["keywords", "wh_phrases"]

    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.question = kwargs['question']
        self.answer_start = kwargs['answer_start']
        self.context = kwargs['context']
        self.answer_text = kwargs['answer_text']
        self.title = kwargs["title"]

        for attr in self.optional_attr:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

    @classmethod
    def from_example_dict(cls, example_dict):
        return cls(**example_dict)

    @classmethod
    def from_dict(cls, val):
        return cls.from_example_dict(val)

    def __repr__(self):
        return repr(vars(self))


class HLCtxtQnAnsTripletEx(CtxtQnAnsTripletEx):
    """
    (Context with Hightlight ans, Question, Answer) example
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = kwargs['highlighted_context']

    @classmethod
    def from_example_dict(cls, example_dict):
        # Need to do high-lighting
        high_lighted_example = highlight_answer(example_dict)
        return cls(**high_lighted_example)

    @classmethod
    def from_dict(cls, val):
        return cls.from_example_dict(val)


class QnGenHLCtxtDataset(Dataset):
    """
    Dataset for question generation with hightlighted ans dataset
    """
    FIELD_DTYPES = {
        "input_ids": np.int16,
        "token_type_ids": np.int8,   # either 0, 1, or 2
        "keyword_pos": np.int16,          # the target word token id
        "wh_phrases_pos": np.int16,
        "question_start_pos": np.int16,
        "num_question_toks": np.int16,
        "num_toks": np.int16,
        "num_keyword_pos": np.int16,
        "num_wh_phrases_pos": np.int16,
    }

    CACHE_ATTRIBUTES = [
        "max_seq_length",
        "doc_stride",
        "max_query_length",
        "train",
        "input_json",
    ]

    def __init__(self, input_json,
                 tokenizer,
                 max_seq_length,
                 doc_stride,
                 max_query_length,
                 processes=NUM_CPUS,
                 train=False,
                 ):

        self.input_json = input_json
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.train = train
        self.processes = min(NUM_CPUS, processes)
        self.verbose = True
        # ------------------
        # Logger
        # ------------------
        self.logger = get_logger(self.__class__.__name__)

        # ------------------
        # Load examples
        # ------------------
        with open(input_json, "r", encoding="utf-8") as f:
            examples = json.load(f)
        if self.train:
            assert "keywords" in examples[0]
            assert "wh_phrases" in examples[0]
        examples = [HLCtxtQnAnsTripletEx.from_dict(e) for e in examples]
        self.examples = examples
        self.logger.info(f"number of examples = {len(examples)}")

        # get features
        self.features = self.conv_examples_to_features()
        self.logger.info("Converted all examples to features")

        self.packed_features = self.pack_features()
        self.logger.info("Packed all features.")

    def __len__(self):
        ret = self.packed_features["num_features"]
        ret = int(ret)
        return ret

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index out of range. {idx} >= {len(self)}")
        # ------------------------------------
        # extract items from packed features
        # ------------------------------------
        # alias
        pked_ftrs = self.packed_features

        # get start idx for flattened array
        if idx == 0:
            s_idx = 0
        else:
            s_idx = pked_ftrs['num_toks'][:idx].sum()

        # id
        id_ = pked_ftrs["id"][idx].decode()
        # num_toks
        num_toks = pked_ftrs["num_toks"][idx]
        # ----------------
        # input tokens
        # ----------------
        input_ids = pked_ftrs["input_ids"][s_idx:s_idx+num_toks]

        # ----------------
        # token_type_ids
        # ----------------
        token_type_ids = pked_ftrs["token_type_ids"][s_idx:s_idx+num_toks]

        # ----------------
        # attention_mask
        # ----------------
        attention_mask = np.full_like(token_type_ids, 1)

        # ----------------
        # question
        # ----------------
        question_start_pos = pked_ftrs["question_start_pos"][idx]
        num_question_toks = pked_ftrs["num_question_toks"][idx]
        # question_ids = input_ids[question_start_pos:-1]

        ret = {
            "id": id_,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "question_start_pos": question_start_pos,
            "num_question_toks": num_question_toks
            # "question_ids": question_ids
        }
        # ---------------------------------------------------------------
        # keywords and wh_phrases for training dataset; Notes: not used.
        if self.train:
            # keywords
            assert "keyword_pos" in pked_ftrs
            assert "num_keyword_pos" in pked_ftrs
            kw_pos_sidx = pked_ftrs['num_keyword_pos'][:idx].sum()
            num_kw_pos = pked_ftrs["num_keyword_pos"][idx]
            ret["keyword_pos"] = pked_ftrs["keyword_pos"][kw_pos_sidx:kw_pos_sidx+num_kw_pos]

            # wh_phrases
            assert "wh_phrases_pos" in pked_ftrs
            assert "num_wh_phrases_pos" in pked_ftrs
            wh_pos_sidx = pked_ftrs['num_wh_phrases_pos'][:idx].sum()
            num_wh_pos = pked_ftrs["num_wh_phrases_pos"][idx]
            ret["wh_phrases_pos"] = pked_ftrs["wh_phrases_pos"][wh_pos_sidx:wh_pos_sidx+num_wh_pos]

        return ret

    def pack_features(self):
        assert hasattr(self, "features")
        assert isinstance(self.features, list)
        n_features = len(self.features)
        packed_features = defaultdict(list)
        for f in self.features:
            for k, v in f.items():
                if k == "id":
                    v = v.encode()
                packed_features[k].append(v)
        # --------------------------------------
        for k, v in packed_features.items():
            # Convert to numpy arrays
            first_val = v[0]
            if isinstance(first_val, list):
                arr = np.concatenate(v, axis=0)
            elif isinstance(first_val, (int, str, bytes, np.number)):
                arr = np.array(v)

            if k in self.FIELD_DTYPES:
                dtype = self.FIELD_DTYPES[k]
                arr = arr.astype(dtype)

            packed_features[k] = arr
        packed_features["num_features"] = np.uint32(n_features)
        return packed_features

    def cache(self, path):
        assert isinstance(self.packed_features, dict)

        # validate packed data
        self._validation()

        # cache instances attributes
        instn_attr = dict()
        for attr in self.CACHE_ATTRIBUTES:
            instn_attr[attr] = getattr(self, attr)

        # save it down
        np.savez_compressed(path, **instn_attr, **self.packed_features)
        self.logger.info(f"Saved cache to {path}")

    @classmethod
    def from_cache(cls, cache_file: Union[Path, str]):
        if isinstance(cache_file, str) and not cache_file.endswith(".npz"):
            cache_file = cache_file + ".npz"
        elif isinstance(cache_file, Path) and cache_file.suffix != ".npz":
            cache_file = cache_file.with_suffix(".npz")

        cache_data = dict()
        with np.load(cache_file) as npzfile:
            for field in npzfile.files:
                cache_data[field] = npzfile[field]
        # ---------------------------------------------
        # create a new instance.
        obj = cls.__new__(cls)
        # initialization empty object
        # Don't forget to call any polymorphic base class initializers
        super(QnGenHLCtxtDataset, obj).__init__()
        obj.input_json = str(cache_data.pop("input_json"))
        obj.max_seq_length = int(cache_data.pop("max_seq_length"))
        obj.doc_stride = int(cache_data.pop("doc_stride"))
        obj.max_query_length = int(cache_data.pop("max_query_length"))
        obj.train = bool(cache_data.pop("train"))
        logger = get_logger(obj.__class__.__name__)
        logger.info("-------- Loaded Attributes --------")
        for k, v in vars(obj).items():
            logger.info(f"{k:17s} : {str(v)}")
        logger.info("-------- Loaded Attributes --------")

        obj.packed_features = cache_data
        obj.logger = logger
        obj.logger.info(f"Loaded from cache data : {cache_file}")
        return obj

    def _validation(self):
        """
        Check across `self.packed_features` and `self.feature`
        """
        assert hasattr(self, "features")
        assert hasattr(self, "packed_features")
        num_features = len(self.features)
        for i in range(num_features):
            f = self.features[i]
            f_from_pked = self[i]
            for k, v in f_from_pked.items():
                oth = f[k]  # get the array from feature
                assert np.array_equal(v, oth)
        self.logger.info("Validated all packed features.")

    def conv_examples_to_features(self):
        if self.processes <= 1:
            ret = []
            for e in tqdm(self.examples, desc="Converting examples to features"):
                ret.append(self.convert_example_to_feature(e))
            return ret

        # i.e. self.processes > 1
        with mp.Pool(self.processes) as p:
            # The process function
            fun = self.convert_example_to_feature
            chunksize = int(len(self.examples) / self.processes / 4)
            chunksize = max(chunksize, 1000)
            self.logger.info(f"Multi-processes chunksize = {chunksize}")
            mapper = p.imap(fun, self.examples, chunksize=chunksize)

            if self.verbose:
                total = len(self.examples)
                desc = "Converting examples to features"
                ret = list(tqdm(mapper, total=total, desc=desc))
            else:
                ret = list(mapper)
        return ret

    def convert_example_to_feature(self, example):
        """
        Convert one example to one feature
        """
        # alias
        tokenizer = self.tokenizer

        # calculate number of special tokens
        n_special_toks = (tokenizer.max_len
                          - tokenizer.max_len_sentences_pair)

        # tokenize
        ctxt_toks = tokenizer.tokenize(example.context)
        qn_toks = tokenizer.tokenize(example.question)
        num_qn_toks = len(qn_toks)
        if num_qn_toks > self.max_query_length:
            qn_toks = list(qn_toks)[:self.max_query_length]
            num_qn_toks = self.max_query_length

        # keyword tokenization
        if self.train:
            keyword_toks = [tokenizer.tokenize(k) for k in example.keywords]
            # filter keywords
            keyword_toks = [
                toks for toks in keyword_toks
                if len(toks) > 0 and tokenizer.unk_token not in toks
            ]
        else:
            keyword_toks = []

        # the maximum number of tokens for context
        max_ctxt_toks = self.max_seq_length - len(qn_toks) - n_special_toks
        num_ctxt_toks = len(ctxt_toks)
        if num_ctxt_toks > max_ctxt_toks:
            # be careful to the args positions
            span = self._select_context_docspan(
                example, ctxt_toks, keyword_toks, max_ctxt_toks)
            ctxt_toks = list(ctxt_toks)[span.start:span.stop]
            num_ctxt_toks = len(ctxt_toks)
        # -----------------------------------------------------------------------
        # create feature dict
        ret = tokenizer.encode_plus(
            ctxt_toks,
            qn_toks,
            max_length=self.max_seq_length,
            return_token_type_ids=True,
            truncation=True,
            truncation_strategy="only_first",
            # all are 1, will contruct attention_mask in __getitem__
            # this ret value just for validation
            return_attention_mask=True,
            # no padding
            pad_to_max_length=False,
        )
        input_ids = ret["input_ids"]
        assert input_ids.index(tokenizer.sep_token_id) == num_ctxt_toks + 1

        # check keyword and wh-phrases pos
        if self.train:
            keyword_pos = []
            for kw in keyword_toks:
                kw_ids = tokenizer.convert_tokens_to_ids(kw)
                keyword_pos.extend(find_idx_of_span_in(input_ids, kw_ids))

            # filter keywords in question
            # keyword_pos = [i for i in keyword_pos if i < num_ctxt_toks+1]
            # sort keywords
            keyword_pos = sorted(keyword_pos)

            wh_pos = []
            wh_toks = [tokenizer.tokenize(k) for k in example.wh_phrases]
            for wh in wh_toks:
                wh_ids = tokenizer.convert_tokens_to_ids(wh)
                wh_pos.extend(find_idx_of_span_in(input_ids, wh_ids))
            wh_pos = sorted(wh_pos)

            ret["keyword_pos"] = keyword_pos
            ret["wh_phrases_pos"] = wh_pos
            ret["num_keyword_pos"] = len(keyword_pos)
            ret["num_wh_phrases_pos"] = len(wh_pos)

        ret["id"] = example.id
        ret["question_start_pos"] = num_ctxt_toks + 2
        ret["num_question_toks"] = num_qn_toks
        ret["num_toks"] = len(input_ids)
        return ret

    def _select_context_docspan(self, example, context_toks, keyword_toks, max_ctxt_toks):
        """
        Select the context that contains the most number of keywords
        """
        # -------------------------
        # select span
        # -------------------------
        hl_tok = self.tokenizer.hl_token

        # check answer start and end
        ans_start, ans_end = get_hl_ans_span(context_toks, hl_tok)
        # -------------------------
        # check keyword end
        kw_idxs = []
        for kw_span in keyword_toks:
            kw_idxs.append(find_idx_of_span_in(context_toks, kw_span))
        flat_kw_idx = [i for kw_idx in kw_idxs for i in kw_idx]
        if flat_kw_idx:
            kw_end = max(flat_kw_idx)
            kw_start = min(flat_kw_idx)
        else:
            kw_start = -100
            kw_end = -100
        # -------------------------
        # create docspan
        docspans = create_docspans(
            len(context_toks), max_ctxt_toks, self.doc_stride)

        # -----------------------------------------------------------------
        # select docspan with only answer
        ans_spans = list()
        for span in docspans:
            if span.start <= ans_start and ans_end <= span.stop:
                ans_spans.append(span)

        if len(ans_spans) == 0:
            raise ValueError("Cannot find any answer span")

        if len(ans_spans) == 1:
            return ans_spans[0]
        # -----------------------------------------------------------------
        if kw_start < 0 or kw_end < 0 or not self.train:
            # cannot find / use any keywords, just return the middle one
            return ans_spans[len(ans_spans) // 2]
        # -----------------------------------------------------------------

        # As we have multiple keywords, just vote the span with most keywords
        votes = list()
        for span in ans_spans:
            vote = 0
            for kw_idx in kw_idxs:
                if kw_idx and span.start <= kw_idx[0] <= span.stop:
                    vote += 1
            votes.append(vote)
        argmax = np.argmax(votes)
        return ans_spans[argmax]
