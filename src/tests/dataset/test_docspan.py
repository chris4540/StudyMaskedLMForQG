import pytest
from utils.data.datasets.squad import CtxtQnAnsTripletEx as CQAWithHLAnsExample
from models.tokenizer import BertTokenizerWithHLAns
from utils.data.datasets.doc_span import create_docspans


class TestConfig:
    max_seq_length: int = 300
    doc_stride: int = 128
    max_query_length: int = 64


@pytest.fixture(scope="module")
def example():
    ret = {
        "id": "572816beff5b5019007d9ce4",
        "title": "Northwestern_University",
        "question": "Which graduate of The Feinburg School of Medicine was the Roswell Park Cancer Institute named after?",
        "answer_start": 142,
        "answer_text": "Mary Harris Thompson",
        "context": "The Feinberg School of Medicine (previously the Northwestern University Medical School) has produced a number of notable graduates, including Mary Harris Thompson, Class of 1870, ad eundem, first female surgeon in Chicago, first female surgeon at Cook County Hospital, and founder of the Mary Thomson Hospital, Roswell Park, Class of 1876, prominent surgeon for whom the Roswell Park Cancer Institute in Buffalo, New York, is named, Daniel Hale Williams, Class of 1883, performed the first successful American open heart surgery; only black charter member of the American College of Surgeons, Charles Horace Mayo, Class of 1888, co-founder of Mayo Clinic, Carlos Montezuma, Class of 1889, one of the first Native Americans to receive a Doctor of Medicine degree from any school, and founder of the Society of American Indians, Howard T. Ricketts, Class of 1897, who discovered bacteria of the genus Rickettsia, and identified the cause and methods of transmission of rocky mountain spotted fever, Allen B. Kanavel, Class of 1899, founder, regent, and president of the American College of Surgeons, internationally recognized as founder of modern hand and peripheral nerve surgery, Robert F. Furchgott, Class of 1940, received a Lasker Award in 1996 and the 1998 Nobel Prize in Physiology or Medicine for his co-discovery of nitric oxide, Thomas E. Starzl, Class of 1952, performed the first successful liver transplant in 1967 and received the National Medal of Science in 2004 and a Lasker Award in 2012, Joseph P. Kerwin, first physician in space, flew on three skylab missions and later served as director of Space and Life Sciences at NASA, C. Richard Schlegel, Class of 1972, developed the dominant patent for a vaccine against human papillomavirus (administered as Gardasil) to prevent cervical cancer, David J. Skorton, Class of 1974, a noted cardiologist became president of Cornell University in 2006, and Andrew E. Senyei, Class of 1979, inventor, venture capitalist, and entrepreneur, founder of biotech and genetics companies, and a university trustee."
    }
    return CQAWithHLAnsExample.from_dict(ret)


@pytest.fixture(scope="module")
def tokenizer():
    modelpath = "bert-base-uncased"
    tokenizer = BertTokenizerWithHLAns.from_pretrained(modelpath)
    return tokenizer


def test_create_doc_spans():
    doc_spans = create_docspans(
        n_tokens=1000, span_max_tokens=512, doc_stride=200)

    # the first one
    assert doc_spans[0].start == 0
    assert doc_spans[0].length == 512

    # second
    assert doc_spans[1].start == 200
    assert doc_spans[1].length == 512

    # third
    assert doc_spans[2].start == 400
    assert doc_spans[2].length == 512

    # the last one
    assert doc_spans[3].start == 600
    assert doc_spans[3].stop == 1000
    assert doc_spans[3].length == 400

    #
    assert doc_spans[0].slice == slice(0, 512)
    assert doc_spans[1].slice == slice(200, 712)
    assert doc_spans[2].slice == slice(400, 912)
    assert doc_spans[3].slice == slice(600, 1000)


def test_doc_span_with_tokenizations(tokenizer, example):
    # ------------------------------------
    cfg = TestConfig()
    max_seq_length = cfg.max_seq_length
    doc_stride = cfg.doc_stride
    max_query_length = cfg.max_query_length
    # ------------------------------------

    # get the tokens
    context_tokens = tokenizer.tokenize(example.context)
    question_tokens = tokenizer.tokenize(example.question)

    # otherwise we have to chop it
    assert len(question_tokens) <= max_query_length

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_ntokens_context = max_seq_length - len(question_tokens) - 3

    doc_spans = create_docspans(len(context_tokens),
                                max_ntokens_context, doc_stride)
    for s in doc_spans:
        span_doc_tokens = context_tokens[s.slice]
        encoded_dict = tokenizer.encode_plus(
            span_doc_tokens,
            question_tokens,
            add_special_tokens=True,
            padding=False,
            truncation="only_first",
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=False,
            # stride=doc_stride,
            stride=max_ntokens_context - doc_stride,
            return_token_type_ids=True,
            # no padding, all are 1, contruct it in dataloader
            return_attention_mask=False
        )

        input_ids = encoded_dict["input_ids"]

        # -------------------------
        # test context
        # -------------------------
        retrived_context_tok_ids = input_ids[1:1+s.length]
        span_doc_tok_ids = tokenizer.convert_tokens_to_ids(span_doc_tokens)

        # test if the lengths equal
        assert len(retrived_context_tok_ids) == len(span_doc_tok_ids)

        # test if all tokens are equal
        for a, b in zip(retrived_context_tok_ids, span_doc_tok_ids):
            assert a == b

        # -------------------------
        # test questions
        # -------------------------
        retrived_question_tok_ids = input_ids[1+s.length+1:-1]  # <---- be careful
        question_tok_ids = tokenizer.convert_tokens_to_ids(question_tokens)

        # test if the lengths equal
        assert len(retrived_question_tok_ids) == len(question_tok_ids)

        # test if all tokens are equal
        for a, b in zip(retrived_question_tok_ids, question_tok_ids):
            assert a == b
