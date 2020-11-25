from models.tokenizer import BertTokenizerWithHLAns
import pytest


@pytest.fixture(scope="module")
def bert_base_vocab():
    toks = BertTokenizerWithHLAns.from_pretrained("bert-base-uncased")
    vocab = toks.vocab
    return vocab


@pytest.mark.parametrize(
    "model_names", [
        "bert-tiny-uncased",
        "bert-mini-uncased",
        "bert-small-uncased",
        "bert-medium-uncased",
    ]
)
def test_bert_tokenzier_vocabs(bert_base_vocab, model_names):
    tokenizer = BertTokenizerWithHLAns.from_pretrained(model_names)
    vocab = tokenizer.vocab
    # check key equals
    bert_base_vocab == vocab

    # check value equals
    for k in bert_base_vocab.keys():
        assert vocab[k] == bert_base_vocab[k]
