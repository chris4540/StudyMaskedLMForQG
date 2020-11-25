import pytest
from utils.data.datasets.wrapper import uPMLMCondTextGenEvalDatasetWrapper
from utils.data.datasets.squad import QnGenHLCtxtDataset
from models.tokenizer import BertTokenizerWithHLAns
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def tokenizer():
    modelpath = "bert-base-uncased"
    tokenizer = BertTokenizerWithHLAns.from_pretrained(modelpath)
    return tokenizer


@pytest.fixture(scope="module")
def dataset(tokenizer):
    # the vscode pytest is hard to config path, use relative path from this
    # test file
    input_json = Path(__file__).parent / "../mock_data/smpl_para_73k_dev.json"
    ds = QnGenHLCtxtDataset(
        input_json=input_json,
        tokenizer=tokenizer,
        max_seq_length=384,
        max_query_length=30,
        doc_stride=128,
        processes=4
    )
    return ds


def test_uPMLMCondTextGenEvalDatasetWrapper(dataset, tokenizer):
    wrapped_ds = uPMLMCondTextGenEvalDatasetWrapper(dataset, tokenizer, sample_decode_length=False)
    for inputs in wrapped_ds:
        attention_mask = inputs["attention_mask"]
        assert np.all(attention_mask)
        input_ids = inputs["input_ids"]

        question = inputs["question"]
        assert question[-1] == tokenizer.sep_token_id
        question_len = len(question) - 1

        assert question_len == inputs["question_len"]

        qstart, qend = np.where(input_ids == 102)[0]
        assert (qend - qstart - 1) == question_len
