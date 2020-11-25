"""
TODO: decompose to smaller classes
"""
import torch
from torch import Tensor
from torch import arange
from typing import List
from .base import BaseTokenDecoder
from .beamsearch_utils import BeamHypotheses
from .retrieval import RetrieveRandomGenerationMinix
from .retrieval import RetrieveCausalGenerationMinix
from .utils import NextTokenCalculationMixin
from .utils import PostProcessNextTokenScoreMixin
from .beamsearch_utils import BeamSearchHypothesisHandlerMixin
from .beamsearch_utils import CalNextBeamForCausalGenerationMixin
from .beamsearch_utils import CalNextBeamForRandomGenerationMixin
from ._workspace import WorkspaceBuildingMixin
from ._workspace import Workspace
from torch.nn.utils.rnn import pad_sequence


class BaseBeamSearchTokenDecoder(BaseTokenDecoder,
                                 WorkspaceBuildingMixin,
                                 NextTokenCalculationMixin,
                                 PostProcessNextTokenScoreMixin):

    _debug = False

    def __init__(self, *args, num_beams=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_beams = num_beams
        self.num_return_sequences = min(num_beams, self.num_return_sequences)
        self.do_sample = False

    # ###############################
    # Prediction related
    # ###############################
    def _build_prediction_steps(self):
        num_beams = self.num_beams
        start = self._workspace.start
        input_ids = self._workspace.input_ids
        effective_batch_size, max_len = self._workspace.input_ids.shape
        # ----------------------------------------------
        # build prediction orders for each input
        # ----------------------------------------------
        batch_pred_steps: List[Tensor] = self._get_batch_pred_steps()

        # ------------------------------------
        # repeate start and end
        # ------------------------------------
        start = start.repeat_interleave(num_beams)
        end = self._workspace.end
        end = end.repeat_interleave(num_beams)
        self._workspace.start = start
        self._workspace.end = end

        assert len(batch_pred_steps) == input_ids.shape[0]
        pred_steps = pad_sequence(batch_pred_steps, batch_first=True, padding_value=max_len - 1)
        iter_steps = pred_steps.shape[1]

        if self._debug:
            self.logger.debug(f"pred_steps = \n{pred_steps}")
            self.logger.debug(f"iter_steps = {iter_steps}")

        self._workspace.pred_steps = pred_steps
        self._workspace.iter_steps = iter_steps

    def _get_batch_pred_steps(self) -> List[Tensor]:
        raise NotImplementedError("Inherit class has to implement this class")

    def _predict_next_batch_beam(self):
        self._calc_next_token_topk()
        self._cal_next_batch_beam()

    def _update_next_tokens(self):
        # ==============
        # inputs
        # ==============
        input_ids = self._workspace.input_ids
        attention_mask = self._workspace.attention_mask
        batch_size = self._workspace.batch_size
        cursors = self._workspace.cursors
        beam_scores = self._workspace.beam_scores
        num_beams = self.num_beams

        # ====================
        # local variables
        # ====================
        num_sents: int = batch_size * num_beams
        next_batch_beam = self._workspace.next_batch_beam
        # ---------------------------------------------------------------------
        # sanity check / prepare next batch
        # ---------------------------------------------------------------------
        # Unpack 'beam_token_score', 'token_id', 'effective_beam_id'
        beam_scores[:] = beam_scores.new(
            [x.beam_token_score for x in next_batch_beam])
        beam_tokens = input_ids.new([x.token_id for x in next_batch_beam])
        beam_idx = input_ids.new([x.effective_beam_id for x in next_batch_beam])
        if self._debug:
            self.logger.debug(f"beam_scores = {beam_scores}")
            self.logger.debug(f"beam_tokens = {beam_tokens}")
            self.logger.debug(f"beam_idx = {beam_idx}")

        # ----------------------------------
        # Update Inputs by filling out masks
        # ----------------------------------
        # rearrage beams by the sorted score
        input_ids[:, :] = input_ids[beam_idx, :]
        # fill out tokens
        input_ids[arange(num_sents), cursors] = beam_tokens
        # allow to put attention to next token
        attention_mask[arange(num_sents), cursors] = 1
        if self._debug:
            self.log_generated_tokens(debug=True)
            self.log_tensor(attention_mask, "attention_mask", debug=True)

        self._workspace.next_sent_beam = None
        self._clean_up_next_tokens_and_scores()

    def _prepare_returns(self):
        # get generated_hyps from workspace
        generated_hyps: List[BeamHypotheses]
        generated_hyps = self._workspace.generated_hyps

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


class uPMLMBSCondTokenDecoder(BaseBeamSearchTokenDecoder,
                              RetrieveRandomGenerationMinix,
                              CalNextBeamForRandomGenerationMixin,
                              BeamSearchHypothesisHandlerMixin):
    """
    u-PMLM beam search conditional token decoder

    Steps:
        1. Generate one generation order per input

    Notes:
        check uPMLMCondTokenDecoder
    """
    _workspace: Workspace
    _debug: bool = True

    def _get_batch_pred_steps(self) -> List[Tensor]:
        # ===========
        # inputs
        # ===========
        num_beams = self.num_beams
        start = self._workspace.start
        len_ = self._workspace.len_

        # -----------------------
        ret = list()
        for s, n in zip(start, len_):
            ord_ = s + torch.randperm(n)  # type: ignore[call-overload]
            ret.extend([ord_] * num_beams)
        return ret

    def forward(self, inputs):
        self._build_workspace(inputs)
        self._build_prediction_steps()

        # retrive built prediction steps and iteration steps
        pred_steps = self._workspace.pred_steps
        iter_steps = self._workspace.iter_steps

        for i in range(iter_steps):
            # obtain cursors and update cursor
            cursors = pred_steps[:, i]
            self._workspace.cursors = cursors
            if self._debug:
                self.logger.debug(f"Iter {i}: cursors = {cursors}")

            # logits calculation
            self._calc_next_token_scores()
            # postprocessing token scores
            self._postprocess_next_token_scores()
            # predict the next token
            self._predict_next_batch_beam()
            # put predicted `next_batch_beam` to input_ids
            self._update_next_tokens()

        # --------------------
        # wrap up
        # --------------------

        # As beam search for random generation ends together, we collect
        # the beam hypotheses at last
        self._finalize_hypotheses()
        ret = self._prepare_returns()

        self._clean_workspace()

        return ret


class CausalMLMBSCondTokenDecoder(BaseBeamSearchTokenDecoder,
                                  RetrieveCausalGenerationMinix,
                                  CalNextBeamForCausalGenerationMixin,
                                  BeamSearchHypothesisHandlerMixin):

    _workspace: Workspace
    _debug: bool = True

    def _get_batch_pred_steps(self) -> List[Tensor]:
        # ===========
        # inputs
        # ===========
        num_beams = self.num_beams
        start = self._workspace.start
        len_ = self._workspace.len_

        if self._debug:
            self.logger.debug(f"start: {start}")
            self.logger.debug(f"len_: {len_}")

        # -----------------------
        ret = list()
        for s, n in zip(start, len_):
            ord_ = s + torch.arange(n)  # type: ignore[call-overload]
            ret.extend([ord_] * num_beams)
        return ret

    def forward(self, inputs):
        # workspace building
        self._build_workspace(inputs)
        self._build_prediction_steps()

        # retrive built prediction steps and iteration steps
        pred_steps = self._workspace.pred_steps
        iter_steps = self._workspace.iter_steps

        for i in range(iter_steps):
            # obtain cursors and update cursor
            cursors = pred_steps[:, i]
            self._workspace.cursors = cursors
            self._workspace.cur_len = i

            if self._debug:
                self.logger.debug(f"Iter {i}: cursors = {cursors}")

            # logits calculation
            self._calc_next_token_scores()
            # postprocessing token scores
            self._postprocess_next_token_scores()

            # predict the next token in the beam
            self._predict_next_batch_beam()

            # update sentences of batchs status
            self._update_is_batch_finished()

            # early exit
            if self._is_all_batch_finished():
                break

            # put predicted `next_batch_beam` to input_ids
            self._update_next_tokens()

        # --------------------
        # wrap up
        # --------------------
        self._finalize_hypotheses()
        ret = self._prepare_returns()

        self._clean_workspace()

        return ret
