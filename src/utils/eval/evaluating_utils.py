from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from typing import List, Dict


def clean_hypothesis(hyp: str) -> str:
    """
    Remove extra words after the question mark of the generated question

    Parameters
    ----------
    hyp : str
        Hypothesis question

    Returns
    -------
    str
        the cleaned hypothesis question
    """
    _loc = hyp.find("?")
    if _loc == -1:  # cannot find a question mark
        return hyp
    else:
        return hyp[:_loc+1]


def compute_bleu(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """
    Compute the BLEU scores from BLEU-1 to BLEU-4

    Parameters
    ----------
    refs : List[str]
        [description]
    hyps : List[str]
        [description]

    Returns
    -------
    Dict[str, float]
        [description]
    """

    ref_dict = dict()
    hyp_dict = dict()
    scorer = Bleu(4)
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        ref_dict[i] = [ref]
        hyp_dict[i] = [hyp]
    bleu_scores, _ = scorer.compute_score(ref_dict, hyp_dict)
    # we pick from BLEU-1 to BLEU-4 score
    metrics = dict()
    for k, sc in enumerate(bleu_scores):
        metrics[f"bleu_{k+1}"] = sc
    return metrics
