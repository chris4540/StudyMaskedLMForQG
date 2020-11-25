import torch
from torch import Tensor
from torch import arange
from torch.nn import functional as F
from collections import namedtuple
from typing import List
from .base import BaseTokenDecoder
from .utils import Workspace
from .utils import calc_banned_ngram_tokens

BeamSearchNextTokenInfo = namedtuple(
    "BeamSearchNextToken", ['beam_token_score', 'token_id', 'effective_beam_id', 'batch_index'])


class BaseBeamSearchTokenDecoder(BaseTokenDecoder):
    def __init__(self, *args, num_beams=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_beams = num_beams
        self.num_return_sequences = min(num_beams, self.num_return_sequences)
        self.do_sample = False


class BeamHypotheses:
    def __init__(self, num_beams, max_length, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = False
        self.num_beams = num_beams
        self.beams = []  # list of tuple of (scores, hyps)
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class CondCausalMLMBeamSearchTokenDecoder(BaseBeamSearchTokenDecoder):

    _debug = False

    def forward(self, inputs):
        # alias
        model = self.model
        tokenizer = self.tokenizer
        max_decode_len = self.max_decode_len
        num_beams = self.num_beams
        vocab_size = tokenizer.vocab_size
        do_sample = self.do_sample
        length_penalty = self.length_penalty

        # get batch size and the maximum length
        input_ids = inputs["input_ids"]
        batch_size, max_len = input_ids.shape
        device = input_ids.device

        # workspace
        self.__workspace = Workspace()

        # ----------------------------
        # generated hypotheses
        # ----------------------------
        generated_hyps: List[BeamHypotheses]
        generated_hyps = [
            BeamHypotheses(num_beams, max_decode_len,
                           length_penalty=length_penalty)
            for _ in range(batch_size)
        ]

        # done sentences
        done = torch.tensor([False] * batch_size)

        # get our extened tensors for model input
        input_ids = self._repeat_input(input_ids)
        attention_mask = self._repeat_input(inputs["attention_mask"])
        token_type_ids = self._repeat_input(inputs["token_type_ids"])
        # get effective batch_size
        eff_batch_size = input_ids.shape[0]

        # prepare iterating indices
        start = inputs["question_start_pos"]
        # add one axis, repeat to num_beams
        start = start.repeat_interleave(num_beams)
        # Here is the example of cursors
        # cursors = [seq1_idx, ..., seq1_idx, seq2_idx, ..., seq2_idx, ...]
        cursors = start.detach().clone()

        # -----------------------------------------
        # scores for each sentence in the beam
        # -----------------------------------------
        size = (batch_size, num_beams)
        beam_scores = torch.zeros(size, dtype=torch.float, device=device)
        if not do_sample:
            beam_scores[:, 1:] = -1e9
        # beam_scores.shape = (batch_size * num_beams,)
        beam_scores = beam_scores.view(-1)

        # calculate the maximum steps to iterate
        iter_steps = min(max_len - start.min(), max_decode_len)

        # ----------------------------
        # Build workspace
        # ----------------------------
        self.__workspace.generated_hyps = generated_hyps
        self.__workspace.done = done
        self.__workspace.beam_scores = beam_scores
        self.__workspace.batch_size = batch_size
        self.__workspace.input_ids = input_ids
        self.__workspace.start = start
        self.__workspace.cursors = cursors

        for cur_len in range(iter_steps):
            self.__workspace.cur_len = cur_len

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = out[0]

            # scores.shape = (batch_size * num_beams, vocab_size)
            next_tok_logits = logits[arange(eff_batch_size), cursors, :]
            assert next_tok_logits.shape == (batch_size*num_beams, vocab_size)

            # select topk results from the logits
            next_scores, next_tokens = self._get_topks_from(next_tok_logits)

            # collect candidates to update next token
            candidates, update_hyp_info = self._collect_candidates(
                next_scores, next_tokens)

            # update hypotheses
            if update_hyp_info:
                self._update_hypotheses(update_hyp_info)
                # need to check how cur_len used
                self._check_if_batch_done(next_scores)

            # ------------------------------------------------------------------
            # exit if decoding completed
            if all(done):
                break
            # ------------------------------------------------------------------
            # Unpack 'beam_token_score', 'token_id', 'effective_beam_id'
            beam_scores[:] = beam_scores.new(
                [x.beam_token_score for x in candidates])
            beam_tokens = input_ids.new([x.token_id for x in candidates])
            beam_idx = input_ids.new([x.effective_beam_id for x in candidates])

            # ----------------------------------
            # Update Inputs by filling out masks
            # ----------------------------------
            # rearrage beams by the sorted score
            input_ids[:, :] = input_ids[beam_idx, :]
            # fill out tokens
            input_ids[arange(eff_batch_size), cursors] = beam_tokens
            # allow to put attention to next token
            attention_mask[arange(eff_batch_size), cursors] = 1
            # ----------------------------------
            # Update cursors
            # ----------------------------------
            # cur_step: either 1 or 0; if done then 0 else 1
            cur_step = (~done).repeat_interleave(num_beams)
            # cursors[:] = cursors[beam_idx] + cur_step
            cursors[:] = cursors + cur_step

            if self._debug:
                print("input_ids id = ", id(input_ids))
                print("start id = ", id(start))
                print("cursors id = ", id(cursors))

        # ------------------------------------
        # Finalize
        # ------------------------------------
        # wrap up uncompleted work
        self._wrap_up_uncompleted_hypotheses()

        # ------------------
        ret = self._prepare_returns()
        # clean up workspace
        self.__workspace = None
        return ret

    def _wrap_up_uncompleted_hypotheses(self):
        """
        Fill out opened hypotheses

        Accessed instance attributes
        --------------------
        __workspace.generated_hyps
        __workspace.beam_scores
        __workspace.batch_size
        num_beams
        """
        num_beams = self.num_beams
        beam_scores = self.__workspace.beam_scores
        batch_size = self.__workspace.batch_size
        done = self.__workspace.done
        generated_hyps = self.__workspace.generated_hyps
        #
        start = self.__workspace.start
        cursors = self.__workspace.cursors
        input_ids = self.__workspace.input_ids

        for b_idx in range(batch_size):
            if done[b_idx]:
                continue
            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                eff_beam_id = b_idx * num_beams + beam_id
                score = beam_scores[eff_beam_id].item()
                # start and end index
                _sidx = start[eff_beam_id]
                _eidx = cursors[eff_beam_id] + 1

                # select output tokens
                out_toks = input_ids[eff_beam_id][_sidx:_eidx]
                generated_hyps[b_idx].add(out_toks, score)

    def _postprocess_next_token_scores(self, scores):
        no_repeat_ngram_size = self.no_repeat_ngram_size
        num_beams = self.num_beams
        batch_size = self.__workspace.batch_size
        input_ids = self.__workspace.input_ids
        start = self.__workspace.start
        cursors = self.__workspace.cursors
        cur_len = self.__workspace.cur_len
        eff_batch_size = input_ids.shape[0]

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            gen_input_ids = []
            for i in range(eff_batch_size):
                gen_input_ids.append(input_ids[i, start[i]:cursors[i]])

            banned_batch_tokens = calc_banned_ngram_tokens(
                gen_input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")
        return scores

    def _prepare_returns(self):
        """
        Prepare return for decoding

        Accessed instance attributes
        --------------------
        __workspace.generated_hyps
        num_ret_seqs
        do_sample

        Return
        ---------------
        the decoded sentences for each batch
        """
        # get generated_hyps from workspace
        generated_hyps: List[BeamHypotheses]
        generated_hyps = self.__workspace.generated_hyps

        if self.do_sample:
            raise NotImplementedError("Sampling is not implemented yet!")
        else:
            num_ret_seq_per_batch = self.num_return_sequences

        # retrieve best hypotheses for each batch
        ret_hyps = []
        ret_scores = []
        for batch_hyp in generated_hyps:
            batch_ret_hyps = []
            batch_ret_scores = []
            sorted_beams = sorted(batch_hyp.beams, key=lambda x: x[0])
            # obtain the first `num_ret_seq_per_batch` hypotheses
            for j in range(num_ret_seq_per_batch):
                beam = sorted_beams.pop()
                score, hyp = beam  # beam = (score, hyp)
                batch_ret_hyps.append(hyp)
                batch_ret_scores.append(score)
            # put batch_ret_* into return
            ret_hyps.append(batch_ret_hyps)
            ret_scores.append(batch_ret_scores)
        return ret_hyps, ret_scores

    def _get_topks_from(self, logits):
        """
        Get top k next tokens and the corresponding scores from logits

        Accessed instance attributes
        --------------------
        num_beams
        tokenizer
        beam_scores
        _debug

        Return
        ---------------
        the top `num_beams` next tokens and the corresponding scores
        """
        # get dimensions
        vocab_size = self.tokenizer.vocab_size
        num_beams = self.num_beams
        batch_size = logits.shape[0] // num_beams

        # beam_scores
        beam_scores = self.__workspace.beam_scores

        # renormalize logits
        scores = F.log_softmax(logits, dim=-1)

        # post process the score
        scores = self._postprocess_next_token_scores(scores)

        # prepare score for getting topk
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)
        # find top 2*num_beams
        next_scores, next_tokens = torch.topk(
            next_scores, 2*num_beams, dim=1, largest=True, sorted=True)

        if self._debug:
            assert scores.shape == (batch_size * num_beams, vocab_size)
            assert next_scores.size() == (batch_size, 2 * num_beams)
            assert next_tokens.size() == (batch_size, 2 * num_beams)
        return next_scores, next_tokens

    def _update_hypotheses(self, update_info: List[BeamSearchNextTokenInfo]):

        generated_hyps = self.__workspace.generated_hyps
        start = self.__workspace.start
        cursors = self.__workspace.cursors
        input_ids = self.__workspace.input_ids

        for info in update_info:
            eff_beam_id = info.effective_beam_id
            bs_score = info.beam_token_score
            batch_index = info.batch_index
            token_id = info.token_id
            assert isinstance(batch_index, int)
            assert batch_index >= 0
            # -------------------------
            # Update the hypothesis
            # -------------------------
            # 1. Get the almost completed result
            out_toks = input_ids[eff_beam_id].detach().clone()
            # 2. Add the last token to the result
            out_toks[cursors[eff_beam_id]] = token_id
            # 3. Trim down
            _sidx = start[eff_beam_id]
            _eidx = cursors[eff_beam_id] + 1
            # start is okay as we just alternate orders within batch
            out_toks = out_toks[_sidx:_eidx]
            # 3. Update the hypothesis list
            generated_hyps[batch_index].add(out_toks, bs_score.item())

    def _check_if_batch_done(self, next_scores):
        generated_hyps = self.__workspace.generated_hyps
        done = self.__workspace.done
        cur_len = self.__workspace.cur_len

        for b_idx, hyp in enumerate(generated_hyps):
            # Check if we are done with this batch
            batch_best_score = float(next_scores[b_idx].max())
            # |= is the OR-assignment operater
            done[b_idx] |= generated_hyps[b_idx].is_done(
                batch_best_score, cur_len)

    def _collect_candidates(self, next_scores, next_tokens):
        """
        Returns
        -------
        candidates: for updating the inputs
        hyp_to_save: for saving
        """
        # obtain dimension
        num_beams = self.num_beams
        vocab_size = self.tokenizer.vocab_size
        batch_size = next_scores.shape[0]

        # obtain variables from workspace
        done = self.__workspace.done
        generated_hyps = self.__workspace.generated_hyps

        # token
        pad_token_id = self.model.config.pad_token_id
        # eos_token_id = self.eos_token_id
        eos_token_id = self.tokenizer.sep_token_id
        # ----------------------------------------------------------------------
        # collect candidates to update next token
        # ------------------------------------------------------------------
        # candidates: candidates of next tokens
        candidates = list()
        hyp_to_save = list()
        for b_idx in range(batch_size):
            # if we are done with this sentence
            if done[b_idx]:
                if len(generated_hyps[b_idx]) < num_beams:
                    raise RuntimeError(
                        f"Batch is done if we have generated at least {num_beams} beams!"
                    )
                # dummy candidate
                dummy_cand = BeamSearchNextTokenInfo(beam_token_score=0,
                                                     token_id=pad_token_id,
                                                     effective_beam_id=0,
                                                     batch_index=b_idx)
                candidates.extend([dummy_cand] * num_beams)
                continue

            # candidates for this batch
            batch_candidates = []
            # next tokens for this sentence
            for r in range(2 * num_beams):  # r = beam_token_rank
                bs_token_id = next_tokens[b_idx, r]
                bs_score = next_scores[b_idx, r]
                # get beam and token IDs
                beam_id = bs_token_id // vocab_size
                token_id = bs_token_id % vocab_size
                eff_beam_id = b_idx * num_beams + beam_id  # effective beam id

                # add to generated hypotheses if end of sentence
                if (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens,
                    # it should not be added
                    if r >= num_beams:
                        continue
                    hyp_to_save.append(
                        BeamSearchNextTokenInfo(
                            beam_token_score=bs_score,
                            token_id=token_id,
                            effective_beam_id=eff_beam_id,
                            batch_index=b_idx))
                else:
                    # add next predicted token since it is not eos_token
                    batch_candidates.append(
                        BeamSearchNextTokenInfo(
                            beam_token_score=bs_score,
                            token_id=token_id,
                            effective_beam_id=eff_beam_id,
                            batch_index=b_idx))

                # once the beam for next step of this batch is full,
                # don't add more tokens to it.
                if len(batch_candidates) == num_beams:
                    break

            # put batch_candidates into candidates
            assert len(batch_candidates) == num_beams
            candidates.extend(batch_candidates)
            assert len(candidates) == num_beams * (b_idx + 1)

        # prepare next batch
        assert len(candidates) == batch_size * num_beams
        return candidates, hyp_to_save

    def _repeat_input(self, val: Tensor) -> Tensor:
        """
        Expand and repeat tensor for bean searching.

        Example
        -------
        >>> decoder = CondMLMCausalBeamSearchDecoder(..., num_beams=3)
        >>> val = [[--------seq 1--------],
                   [--------seq 2--------]]
        >>> x = decoder._repeat_input(val)
        >>> x
        [[--------seq 1--------],
         [--------seq 1--------],
         [--------seq 1--------],
         [--------seq 2--------],
         [--------seq 2--------],
         [--------seq 2--------]]
        """
        batch_size, val_len = val.shape
        num_beams = self.num_beams
        num_ret_seqs = self.num_return_sequences
        do_sample = self.do_sample

        # --------------------------
        # Set:
        #   1. effective batch size,
        #   2. and effective batch multiplier
        # according to do_sample
        # --------------------------
        if do_sample:
            eff_batch_size = batch_size * num_ret_seqs
            eff_batch_mult = num_ret_seqs
        else:
            eff_batch_size = batch_size
            eff_batch_mult = 1

        # -----------------------------
        # Reshape the input tensor
        # -----------------------------
        # expand val tensor from [batch_size, val_len] to
        # [batch_size, num_beams, val_len]
        ret = val.unsqueeze(1).expand(
            batch_size, eff_batch_mult * num_beams, val_len)

        # collepse it into [ (batch_size * num_return_sequences * num_beams, val_len)]
        ret = ret.contiguous().view(eff_batch_size * num_beams, val_len)

        return ret
