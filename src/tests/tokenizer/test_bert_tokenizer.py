import pytest
from models.tokenizer import BertTokenizerWithHLAns
from transformers import BertConfig


@pytest.fixture(scope="module")
def tokenizer():
    modelpath = "bert-base-uncased"
    ret = BertTokenizerWithHLAns.from_pretrained(modelpath)
    return ret

# ---------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------


def test_HL_in_special_tokens(tokenizer):
    assert "[HL]" in tokenizer.all_special_tokens


def test_tokenization(tokenizer):
    text = ("Dummy. although he had already eaten a large meal"
            "he was still very [HL]hungry[HL].")
    tokenized_text = tokenizer.tokenize(text)
    assert "[HL]" in tokenized_text


def test_get_token(tokenizer):
    assert tokenizer.hl_token == "[HL]"


def test_get_token_id(tokenizer):
    assert tokenizer.hl_token_id > 0


def test_vocab_size(tokenizer):
    cfg = BertConfig()
    cfg.vocab_size == len(tokenizer)


def test_segment_ids(tokenizer):
    """
    Test the segment_ids (token_type_id)

    Test only the encode_plus method
    """
    # simple encoding
    seq_a = "This is a short sequence."
    encode_dict = tokenizer.encode_plus(
        seq_a, max_length=20, pad_to_max_length=True)

    assert all(v == 0 for v in encode_dict['token_type_ids'])
    assert sum(encode_dict['attention_mask']) == 6 + 2
    # ----------------------------------------------------------------
    seq_a = "This is a [HL]short[HL] sequence."
    seq_b = "How is the sequence?"
    encode_dict = tokenizer.encode_plus(
        seq_a, seq_b, max_length=20, pad_to_max_length=True)

    # [CLS] x 1
    # [HL] x 2
    # [SEP] x 2
    assert sum(encode_dict['attention_mask']) == 16

    ans = get_ans_tokens_from(tokenizer, encode_dict)

    assert ans == ['[HL]', 'short', '[HL]']


def test_partial_segment_ids(tokenizer):
    seq_a = "This is a [HL]long[HL] sequence."
    seq_b = "How is the sequence?"
    encode_dict = tokenizer.encode_plus(
        seq_a, seq_b, max_length=12, pad_to_max_length=True)

    ans = get_ans_tokens_from(tokenizer, encode_dict)

    assert ans == ['[HL]', 'long']


@pytest.mark.parametrize("return_tensors", ['tf', 'pt'])
def test_torch_tensor_support(tokenizer, return_tensors):
    context_text = (
        "Due to Verizon Communications exclusivity, "
        "streaming on smartphones was only provided to Verizon Wireless customers "
        "via the NFL Mobile service.")
    question = "Which wireless company had exclusive streaming rights on [MASK]"

    print(return_tensors)
    # TODO: assertion
    tokenizer.encode(
        context_text + question, pad_to_max_length=False, return_tensors=return_tensors)


def get_ans_tokens_from(tokenizer, encode_dict):
    ret = list()

    input_ids = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']

    # convert input_ids to text tokens
    txt_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    for txt, token_type in zip(txt_tokens, token_type_ids):
        if token_type == 2:
            ret.append(txt)

    return ret
