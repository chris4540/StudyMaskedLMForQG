import torch
from torch import Tensor
from torch import IntTensor
from torch.nn import functional as F
from typing import List, Dict, Tuple, Union, Callable
from collections import defaultdict
from ._workspace import Workspace


def ngrams(tokens, n):
    """
    Reference
    ---------
    http://www.locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    """
    if n < 1:
        raise ValueError("The number of grams should be >= 1.")
    return zip(*[tokens[i:] for i in range(n)])


# ===============================================
# Calculate which ngram tokens to be banned
# ===============================================
def build_generated_ngram_dicts(
        generated_toks: Union[List[Tensor], Tensor],
        no_repeat_ngram_size: int) -> List[Dict[Tuple, List[int]]]:
    """
    Given the generated tokens per hypothesis, statistic a (ngram minus one)
    dictionary per hypothesis. The return is a list of used ngram dictionary
    called `ngrams_dict`.

    Take when we have only one hypotheses as an example

    Example
    -----------
    Generated tokens of one hypothesis:
        ["what", "do", "you", "do", "he", "do", "she"]
    no_repeat_ngram_size:
        2

    ngrams_dict to be returned:
        {
            "('what',)": ["do"],
            "('do',)": ["you", "he", "she"],
            "('you',)": ["do"],
            "('he',)": ["do"],
            "('she',)": [],
        }

    Return
    ------
    ngram_dicts: a list of ngrams_dict
        ngrams_dict: for ngram, use first (n-1)th-words as key; nth-word as value
        ngrams_dict[(w1, w2, ..., w_{n-1})] = [w_{n}]
    """
    # --------------------------------------------
    # get the number of hypotheses `num_hypos`
    # --------------------------------------------
    try:
        # consider `generated_toks` as a tensor
        num_hypos = generated_toks.shape[0]  # type: ignore
    except AttributeError:
        assert isinstance(generated_toks, list)
        num_hypos = len(generated_toks)

    # -----------------
    # Build return
    # -----------------
    ngram_dicts: List[Dict[Tuple, List[int]]]
    ngram_dicts = [defaultdict(list) for _ in range(num_hypos)]

    for i in range(num_hypos):
        # toks: Generated tokens
        toks = generated_toks[i].tolist()
        # ngram_dict: a dictionary of ngrams
        ngram_dict = ngram_dicts[i]
        for ng in ngrams(toks, no_repeat_ngram_size):
            ngram_dict[tuple(ng[:-1])] += [ng[-1]]

    return ngram_dicts


def calc_banned_ngram_tokens(
        prev_input_ids: Union[List[Tensor], Tensor],
        num_hypos: int, no_repeat_ngram_size: int,
        cur_len: int) -> List[List[int]]:
    """
    Copied from fairseq for no_repeat_ngram in beam_search and then modified it.

    This function is only for causal language generation.

    Return
    ----------
    List[List[int]]
        banned_tokens: banned token ids for each batch

    Reference
    --------------
    - transformers.generation_utils.calc_banned_ngram_tokens
    """
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] * num_hypos]

    # ngrams_dict: for ngram, use first (n-1)th-words as key; nth-word as value
    # E.g. ngrams_dict[(w1, w2, ..., w_{n-1})] = w_{n}
    # ngrams_dicts: list of ngrams_dict
    ngram_dicts: List[Dict[Tuple, List[int]]]
    ngram_dicts = build_generated_ngram_dicts(prev_input_ids, no_repeat_ngram_size)
    assert len(ngram_dicts) == num_hypos

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx][start_idx:cur_len].tolist())
        ret = ngram_dicts[hypo_idx][ngram_idx]
        return ret

    banned_tokens = [
        _get_generated_ngrams(i) for i in range(num_hypos)
    ]
    return banned_tokens


class PostProcessNextTokenScoreMixin:
    _workspace: Workspace
    _debug: bool
    # give out the function signature for mypy type checking
    get_gen_input_ids: Callable[[], List[Tensor]]

    def _postprocess_next_token_scores(self):
        no_repeat_ngram_size: int = self.no_repeat_ngram_size

        if no_repeat_ngram_size <= 0:
            return
        # =============
        # inputs
        # =============
        vocab_size = self._workspace.vocab_size
        next_token_scores = self._workspace.next_token_scores
        batch_size = self._workspace.batch_size
        num_beams = self.num_beams

        # ================
        # local variabels
        # ================
        neg_inf: float = -float("inf")

        # ================
        # sanity check
        # ================
        assert next_token_scores.shape == (batch_size*num_beams, vocab_size)

        # As the tokens are generated in random order, give the whole questions
        # to calculate which tokens should be banned
        gen_input_ids: List[Tensor] = self.get_gen_input_ids()
        self._workspace.gen_input_ids = gen_input_ids

        self.__calc_banned_ngram_tokens()

        # apply banned tokens
        batch_banned_tokens = self._workspace.batch_banned_tokens
        for i, t in enumerate(batch_banned_tokens):
            next_token_scores[i, t] = neg_inf

    def __calc_banned_ngram_tokens(self):

        # TODO: consider some exit condition for l2r decoding

        # =============
        # inputs
        # =============
        gen_input_ids: List[Tensor] = self._workspace.gen_input_ids
        input_ids = self._workspace.input_ids
        no_repeat_ngram_size: int = self.no_repeat_ngram_size
        cursors: IntTensor = self._workspace.cursors
        start: IntTensor = self._workspace.start
        num_beams: int = self.num_beams
        batch_size: int = self._workspace.batch_size

        # =================
        # local variables
        # =================
        # num_sents: the number of sentences we are predicting
        num_sents: int = num_beams * batch_size
        # ------------------------------------------------------------------

        # get the ngram-1 dictionary
        ngram_dicts: List[Dict[Tuple, List[int]]]
        ngram_dicts = build_generated_ngram_dicts(
            gen_input_ids, no_repeat_ngram_size)
        assert len(ngram_dicts) == num_sents

        # get banned_tokens
        batch_banned_tokens = list()
        for i in range(num_sents):
            e = cursors[i]
            s = e + 1 - no_repeat_ngram_size
            if s <= start[i]:
                # add empty list
                banned_tok = []
            else:
                ng_prev = tuple(input_ids[i][s:e].tolist())
                assert len(ng_prev) == no_repeat_ngram_size - 1
                banned_tok = ngram_dicts[i][ng_prev]
            batch_banned_tokens.append(banned_tok)

        if self._debug:
            self.logger.debug(f"batch_banned_tokens = {batch_banned_tokens}")

        # =============
        # output
        # =============
        self._workspace.batch_banned_tokens = batch_banned_tokens


class NextTokenCalculationMixin:
    """
    This class provides functions to:
        1. calculate next position logits
        2. logits normalization
        3. calculate top-k for `next_tokens` and `next_scores`
    """
    _workspace: Workspace

    def _calc_next_token_scores(self):
        # ---------------
        # inputs
        # ---------------
        model = self._workspace.model
        input_ids = self._workspace.input_ids
        attention_mask = self._workspace.attention_mask
        token_type_ids = self._workspace.token_type_ids
        cursors = self._workspace.cursors

        # ---------------
        # local variabels
        # ---------------
        num_sents = input_ids.shape[0]

        # ask model to predict
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = out[0]
        next_token_logits = logits[torch.arange(num_sents), cursors, :]
        # renormalize logits
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        self._workspace.next_token_scores = next_token_scores

    def _calc_next_token_topk(self):

        # -----------
        # inputs
        # -----------
        batch_size = self._workspace.batch_size
        vocab_size = self._workspace.vocab_size
        beam_scores = self._workspace.beam_scores
        num_beams = self.num_beams

        # ---------------------
        # Top-k
        # ---------------------
        # prepare score for getting topk
        scores = self._workspace.next_token_scores
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        # move `num_beams` to 2nd dimension; for retreive top-k results among beams
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)
        # find top 2*num_beams
        next_scores, next_tokens = torch.topk(
            next_scores, 2*num_beams, dim=1, largest=True, sorted=True)

        if self._debug:
            assert scores.shape == (batch_size * num_beams, vocab_size)
            assert next_scores.size() == (batch_size, 2 * num_beams)
            assert next_tokens.size() == (batch_size, 2 * num_beams)

        # --------------
        # outputs
        # --------------
        self._workspace.next_scores = next_scores
        self._workspace.next_tokens = next_tokens

    def _clean_up_next_tokens_and_scores(self):
        self._workspace.next_scores = None
        self._workspace.next_tokens = None
