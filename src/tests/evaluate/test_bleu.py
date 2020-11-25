"""
This is the example to show bleu score calculation between nltk and nlgeval
are equivalent with the same smoothing function
"""
from nltk.translate.bleu_score import sentence_bleu
import pytest
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
import numpy as np


def smoothing_fun(p_n, *args, **kwargs):
    """
    Smoothing method from nlg-eval
    """
    small = 1e-9
    tiny = 1e-15  # so that if guess is 0 still return 0
    ret = [
        (p_i.numerator + tiny) / (p_i.denominator + small)
        if p_i.numerator == 0
        else p_i
        for p_i in p_n
    ]
    return ret

# ------------------------------------------------------------------------------
# ref = [1996, 2118, 1005, 1055, 2415, 1999, 7211, 2003, 2284, 2279, 2000, 2054,
#        2082, 1005, 1055, 3721, 1029]
# hyps = [[2054, 2003, 1996, 3721, 1999, 15030, 11692, 2212, 1029],
#         [2054, 2003, 1996, 2171, 1997, 1996, 2118, 1997, 3190, 1029],
#         [2054, 2003, 1996, 2118, 1997, 3190, 1029]]


@pytest.fixture(scope="module")
def sent_ref():
    ref = "the university's center in beijing is located next to what school's campus?"
    return ref


@pytest.fixture(scope="module")
def sent_hyps():
    hyps = [
        "what is the campus in haidian district?",
        "what is the name of the university of chicago?",
        "what is the university of chicago?"
    ]
    return hyps


def test_if_sent_lv_bleu_equivalent(sent_ref, sent_hyps):
    ref_toks = sent_ref.split()
    scorer = Bleu(4)
    for hyp in sent_hyps:
        hyp_toks = hyp.split()
        bleu4_nltk = sentence_bleu(
            [ref_toks], hyp_toks, smoothing_function=smoothing_fun)
        scores, _ = scorer.compute_score({0: [sent_ref]}, {0: [hyp]})
        bleu4_nlgeval = scores[3]
        assert np.allclose(bleu4_nltk, bleu4_nlgeval)
