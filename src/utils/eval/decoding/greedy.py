import torch
from torch import arange
from torch.nn import functional as F
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
from torch.nn.utils.rnn import pad_sequence
from .base import BaseTokenDecoder
from .utils import Workspace
from .utils import calc_banned_ngram_tokens
from .utils import ngrams


class BaseGreedyTokenDecoder(BaseTokenDecoder):

    def _predict_next_token(self):
        """
        This function describe one step prediction in generation loop
        """
        # assert self._workspace is not None

        # retrieve useful stuff from workspace
        model = self._workspace.model
        input_ids = self._workspace.input_ids
        attention_mask = self._workspace.attention_mask
        token_type_ids = self._workspace.token_type_ids
        batch_size = self._workspace.batch_size
        cursors = self._workspace.cursors
        sum_logprobs = self._workspace.sum_logprobs
        finished_sents = self._workspace.finished_sents
        sep_token_id = self._workspace.sep_token_id
        # ---------------------------------------------
        unfinished_sents = ~finished_sents

        # ask model to predict
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = out[0]
        next_token_logits = logits[arange(batch_size), cursors, :]
        # renormalize logits
        next_token_logp = F.log_softmax(next_token_logits, dim=-1)
        next_token_logp = self._postprocess_next_token_scores(next_token_logp)

        # greedy take the best to next
        next_logprob, next_token = torch.max(next_token_logp, dim=-1)

        # update the decoded score
        sum_logprobs += next_logprob*unfinished_sents

        # write back generated token
        toks_to_add = next_token*unfinished_sents + sep_token_id*(finished_sents)
        input_ids[arange(batch_size), cursors] = toks_to_add
        # allow to put attention to next token
        attention_mask[arange(batch_size), cursors] = 1

        # add next_token to workspace for check if need to update
        self._workspace.next_token = next_token
        # print(next_token)

    def _update_sent_state_of_completion(self):
        """
        Update the boolean array telling if the sentence completed generation
        """
        finished_sents = self._workspace.finished_sents
        next_token = self._workspace.next_token
        sep_token_id = self._workspace.sep_token_id
        cursors = self._workspace.cursors
        end = self._workspace.end
        # perform element-wise OR operation
        finished_sents[:] = finished_sents | (next_token == sep_token_id) | (cursors >= end)
        self._workspace.finished_sents = finished_sents

    def _build_workspace(self, inputs):

        # the workspace
        self._workspace = Workspace()

        # model
        self._workspace.model = self.model
        self._workspace.sep_token_id = self.tokenizer.sep_token_id

        # ------------
        # clone inputs
        # ------------
        inputs = self._clone_inputs(inputs)
        # -------------------------------------------------
        # get input tensors from input dict to workspace
        # -------------------------------------------------
        input_ids = inputs["input_ids"]
        self._workspace.input_ids = input_ids
        self._workspace.attention_mask = inputs["attention_mask"]
        self._workspace.token_type_ids = inputs["token_type_ids"]
        # ===================================================================
        # obtain dimension information
        batch_size, max_len = input_ids.shape
        self._workspace.batch_size = batch_size
        # -------------------------
        # iteratiion related
        # -------------------------
        start = inputs["question_start_pos"]
        self._workspace.start = start
        cursors = start.clone()
        self._workspace.cursors = cursors
        # calculate the maximum steps to iterate
        iter_steps = min(max_len - start.min(), self.max_decode_len)
        self._workspace.iter_steps = iter_steps

        device = input_ids.device
        # the sentence score according the logits
        sum_logprobs = torch.zeros((batch_size,), dtype=torch.float, device=device)
        self._workspace.sum_logprobs = sum_logprobs

        # state of each sentence
        finished_sents = torch.full((batch_size,), False, dtype=torch.bool, device=device)
        self._workspace.finished_sents = finished_sents

        # ------------------------------------
        # the end of indices of each sentence
        # ------------------------------------
        end = torch.full_like(start, max_len - 1)
        if self.decode_length_known:
            if "question_len" in inputs:
                len_ = inputs["question_len"]
                self._workspace.len_ = len_
                end = start + len_
                end[end >= max_len] = max_len - 1  # set the upper bound
            else:
                warnings.warn(
                    "`decode_length_known` is True but cannot find "
                    "`question_len` in inputs"
                )
        self._workspace.end = end


class CausalMLMCondTokenDecoder(BaseGreedyTokenDecoder):
    """
    Conditional left-to-right masked language model decoder
    using greedy algorithm
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_return_sequences > 1:
            self.num_return_sequences = 1

    def forward(self, inputs):

        self._build_workspace(inputs)

        cursors = self._workspace.cursors
        iter_steps = self._workspace.iter_steps

        for cur_len in range(iter_steps):
            self._workspace.cur_len = cur_len

            # predict the next token
            self._predict_next_token()

            # update the sentence state
            self._update_sent_state_of_completion()

            # update cursors if not finished
            finished_sents = self._workspace.finished_sents
            cursors[:] += (~finished_sents)

            # if all are finish
            if all(finished_sents):
                break
        # --------------------
        # prepare the return
        # --------------------
        ret = self._prepare_returns()

        # clean up workspace
        self._workspace = None

        return ret

    def _prepare_returns(self):
        batch_size = self._workspace.batch_size
        input_ids = self._workspace.input_ids
        start = self._workspace.start
        cursors = self._workspace.cursors
        sum_logprobs = self._workspace.sum_logprobs
        # retrieve best hypotheses for each batch
        ret_hyps = []
        ret_scores = []
        for b in range(batch_size):
            s_idx = start[b]
            e_idx = cursors[b] + 1  # add one for extra [SEP] token
            hyp = input_ids[b, s_idx:e_idx].tolist()
            # this formula comes from BeamHypotheses
            # which is to normalize the score for different lengths
            score = sum_logprobs[b] / (len(hyp) ** self.length_penalty)
            score = score.item()
            ret_hyps.append([hyp])
            ret_scores.append([score])
        return ret_hyps, ret_scores

    def _postprocess_next_token_scores(self, scores):
        """
        """
        no_repeat_ngram_size = self.no_repeat_ngram_size
        batch_size = self._workspace.batch_size
        input_ids = self._workspace.input_ids
        start = self._workspace.start
        cursors = self._workspace.cursors
        cur_len = self._workspace.cur_len

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            gen_input_ids = []
            for i in range(batch_size):
                gen_input_ids.append(input_ids[i, start[i]:cursors[i]])

            banned_batch_tokens = calc_banned_ngram_tokens(
                gen_input_ids, batch_size, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")
        return scores


class uPMLMCondTokenDecoder(BaseGreedyTokenDecoder):
    """
    TODO: merge similar logic in CausalMLMCondTokenDecoder
    """

    def _update_sent_state_of_completion(self):
        """
        Update the boolean array telling if the sentence completed generation
        """
        finished_sents = self._workspace.finished_sents
        cursors = self._workspace.cursors
        end = self._workspace.end
        # perform element-wise OR operation
        finished_sents[:] = finished_sents | (cursors >= end)
        self._workspace.finished_sents = finished_sents

    def _build_prediction_steps(self):
        """
        We preform random generation
        """
        start = self._workspace.start
        len_ = self._workspace.len_
        max_len = self._workspace.input_ids.shape[1]  # use input_ids as reference
        batch_pred_steps = [s + torch.randperm(n) for s, n in zip(start, len_)]
        pred_steps = pad_sequence(batch_pred_steps, batch_first=True, padding_value=max_len - 1)
        iter_steps = pred_steps.shape[1]
        self._workspace.pred_steps = pred_steps
        self._workspace.iter_steps = iter_steps

    def forward(self, inputs):
        # inputs = self._clone_inputs(inputs)
        self._build_workspace(inputs)

        self._build_prediction_steps()
        pred_steps = self._workspace.pred_steps
        iter_steps = self._workspace.iter_steps

        for i in range(iter_steps):
            # obtain cursors and update cursor
            cursors = pred_steps[:, i]
            # print("cursors", cursors)
            self._workspace.cursors = cursors

            # predict the next token
            self._predict_next_token()

            # update the sentence state
            self._update_sent_state_of_completion()

            # update cursors if not finished
            finished_sents = self._workspace.finished_sents
            # if all are finish
            if all(finished_sents):
                break

        # --------------------
        # prepare the return
        # --------------------
        ret = self._prepare_returns()

        # clean up workspace
        self._workspace = None
        return ret

    def _prepare_returns(self):
        batch_size = self._workspace.batch_size
        input_ids = self._workspace.input_ids
        start = self._workspace.start
        sep_token_id = self._workspace.sep_token_id
        end = self._workspace.end
        sum_logprobs = self._workspace.sum_logprobs
        # retrieve best hypotheses for each batch
        ret_hyps = []
        ret_scores = []
        for b in range(batch_size):
            s_idx = start[b]
            e_idx = end[b] + 1  # add one for extra [SEP] token
            hyp = input_ids[b, s_idx:e_idx].tolist()
            assert hyp[-1] == sep_token_id
            # this formula comes from BeamHypotheses
            # which is to normalize the score for different lengths
            score = sum_logprobs[b] / (len(hyp) ** self.length_penalty)
            score = score.item()
            ret_hyps.append([hyp])
            ret_scores.append([score])
        return ret_hyps, ret_scores

    def _postprocess_next_token_scores(self, scores):
        no_repeat_ngram_size = self.no_repeat_ngram_size

        if no_repeat_ngram_size > 0:
            banned_batch_tokens = self.calc_banned_ngram_tokens()
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")
        return scores

    def calc_banned_ngram_tokens(self):
        input_ids = self._workspace.input_ids
        batch_size = self._workspace.batch_size
        no_repeat_ngram_size = self.no_repeat_ngram_size
        cursors = self._workspace.cursors
        start = self._workspace.start
        end = self._workspace.end

        # Select only the generation part to calculate
        prev_input_ids = [
            input_ids[i, start[i]:end[i]]
            for i in range(batch_size)
        ]
        # ngrams_dict: for ngram, use first (n-1)th-words as key; nth-word as value
        # E.g. ngrams_dict[(w1, w2, ..., w_{n-1})] = w_{n}
        # ngrams_dicts: list of ngrams_dict
        ngrams_dicts: List[Dict[Tuple, List[int]]]
        ngrams_dicts = [defaultdict(list) for _ in range(batch_size)]

        for i in range(batch_size):
            gen_tokens = prev_input_ids[i].tolist()
            ngrams_dict = ngrams_dicts[i]
            for ng in ngrams(gen_tokens, no_repeat_ngram_size):
                ngrams_dict[tuple(ng[:-1])] += [ng[-1]]

        # get banned_tokens
        banned_tokens = list()
        for i in range(batch_size):
            e = cursors[i]
            s = e + 1 - no_repeat_ngram_size
            if s <= start[i]:
                # add empty list
                banned_tok = []
            else:
                ng_prev = tuple(input_ids[i][s:e].tolist())
                assert len(ng_prev) == no_repeat_ngram_size - 1
                banned_tok = ngrams_dicts[i][ng_prev]
            banned_tokens.append(banned_tok)
        return banned_tokens
