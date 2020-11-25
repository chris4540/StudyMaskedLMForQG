import pytest
from models.bert_qgen import BertForMaskedLM
from transformers import BertConfig


def test_resize_type_token_embeddings():
    cfg = BertConfig(type_vocab_size=2)
    model = BertForMaskedLM(cfg)
    n_emb = 5
    emb = model.resize_type_token_embeddings(n_emb)

    # check the number of dimension
    assert model.config.type_vocab_size == n_emb
    assert emb.num_embeddings == n_emb

    # check if the same object
    assert id(emb) == id(model.bert.embeddings.token_type_embeddings)


def test_resize_type_token_embeddings_from_pretrain():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    n_emb = 5
    emb = model.resize_type_token_embeddings(n_emb)

    # check the number of dimension
    assert model.config.type_vocab_size == n_emb
    assert emb.num_embeddings == n_emb

    # check if the same object
    assert id(emb) == id(model.bert.embeddings.token_type_embeddings)
