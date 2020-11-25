from models.bert_qgen import BertForMaskedLM
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.squad import QnGenHLCtxtDataset
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from torch.utils.data.dataloader import DataLoader
import torch
from utils.data.datasets.data_collator import DataCollatorForPadding
from utils.eval.decoding import CondCausalMLMBeamSearchTokenDecoder
import pytest
import numpy as np

EvaluationDatasetWrapper = CausalCondTextGenEvalDatasetWrapper


@pytest.fixture(scope="module")
def model():
    model_path = "./tests/mock_data/dev-bert-mini"
    ret = BertForMaskedLM.from_pretrained(model_path)
    ret.eval()
    return ret


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer_path = "google/bert_uncased_L-4_H-256_A-4"
    tokenizer = BertTokenizerWithHLAns.from_pretrained(tokenizer_path)
    return tokenizer


@pytest.fixture(scope="module")
def dataset(tokenizer):
    sample_dev_json = "./tests/mock_data/smpl_para_73k_dev.json"
    _txtds = QnGenHLCtxtDataset(
        input_json=sample_dev_json,
        tokenizer=tokenizer,
        max_seq_length=384,
        max_query_length=30,
        doc_stride=128,
        train=False,
        processes=1,
    )
    ret = EvaluationDatasetWrapper(_txtds, tokenizer)
    return ret


def do_decoding(model, tokenizer, dataloader, token_decoder):
    ret = []
    for inputs in dataloader:
        decoded, score = token_decoder(inputs)
        for i, batch_decoded in enumerate(decoded):
            s = score[i][0]
            sent = batch_decoded[0]
            ret.append((s, sent))
    return ret


def get_dataloader(tokenizer, dataset, batch_size=1):
    pad_collator = DataCollatorForPadding(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collator)
    return dataloader


@pytest.fixture(scope="module")
def bs_tok_decoder(model, tokenizer):
    ret = CondCausalMLMBeamSearchTokenDecoder(
        model, tokenizer, max_decode_len=30, num_return_sequences=1)
    return ret


@pytest.fixture(scope="module")
def batch1_dataloader(tokenizer, dataset):
    ret = get_dataloader(tokenizer, dataset, batch_size=1)
    return ret


@pytest.fixture(scope="module")
def bs_1batch_decoded(model, tokenizer, bs_tok_decoder, batch1_dataloader):
    ret = do_decoding(model, tokenizer, batch1_dataloader, bs_tok_decoder)
    return ret


@pytest.mark.parametrize("batch_size", [2, 3, 5])
def test_different_batch_size(model, tokenizer, dataset, bs_tok_decoder, bs_1batch_decoded, batch_size):
    reference = bs_1batch_decoded
    dataloader = get_dataloader(tokenizer, dataset, batch_size=1)
    results = do_decoding(model, tokenizer, dataloader, bs_tok_decoder)

    assert len(reference) == len(results)
    for ref, res in zip(reference, results):
        assert np.allclose(ref[0], res[0])  # the score
        assert all(torch.eq(ref[1], res[1]))  # the index tensor
