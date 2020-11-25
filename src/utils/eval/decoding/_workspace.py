"""
This module provide a Mixin class `WorkspaceBuildingMixin` to build a workspace

"""
import torch
from torch import Tensor
from torch import IntTensor
from torch import FloatTensor
from torch import BoolTensor
import warnings
import transformers
from typing import List, Dict
from .beamsearch_utils import BeamHypotheses


class Workspace:
    """
    A table to put variables, data, information for decoding (text generation)

    Notes
    -----
    As the original version of text generation is too lengthy and hard to
    reuse, we use a simple strategy to share information when decoding.
    """
    # ---------------------
    # Model
    # ---------------------
    model: transformers.BertForMaskedLM

    # ---------------------
    # bert decoding info
    # ---------------------
    input_ids: IntTensor
    attention_mask: IntTensor
    token_type_ids: IntTensor
    # ---------------
    # Start & end
    # ---------------
    start: IntTensor
    end: IntTensor
    len_: IntTensor
    cur_len: int   # causal generation only
    # ---------------
    # cursor
    # ---------------
    cursors: IntTensor

    # ---------------
    # dimensions
    # ---------------
    batch_size: int
    vocab_size: int
    effective_batch_size: int

    # ---------------------------------------
    # Beam search top-k related
    # ---------------------------------------
    next_scores: FloatTensor
    next_tokens: IntTensor

    # ---------------
    # Mics
    # ---------------
    next_token_scores: FloatTensor
    device: torch.device
    is_batch_finished: BoolTensor


class WorkspaceBuildingMixin:

    # These variables are from the host class
    num_beams: int
    num_return_sequences: int
    do_sample: bool
    _debug: bool

    def _build_workspace(self, inputs):
        if self._debug:
            self.logger.debug("Building workspaces")
        # ---------------
        # Workspace
        # ---------------
        self._workspace = Workspace()

        # ------------
        # model
        # ------------
        self._workspace.model = self.model

        # -------------------------------------------------------
        # get input tensors reference and obtain dim. info
        # -------------------------------------------------------
        input_ids = inputs["input_ids"]
        batch_size, max_len = input_ids.shape
        self._workspace.batch_size = batch_size
        self._workspace.device = input_ids.device

        # -------------------------------------------------------
        # Extend inputs and add to workspace
        # -------------------------------------------------------
        extended_inputs = self.__extend_inputs(inputs)
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            setattr(self._workspace, k, extended_inputs[k])

        # other infos
        self._workspace.vocab_size = self.tokenizer.vocab_size

        # TODO: check if extend the usage to greedy
        # add decode start and end to workspace
        self.__add_decode_start_end_to_wksp(inputs)
        # build beam score
        self.__add_beam_scores_to_wksp()

        # Add generated_hypotheses to workspace as a placeholder
        self.__add_generated_hypotheses_placeholder_to_wksp()

        # add `is_batch_finished` to workspace
        self._add_is_batch_finished_to_wksp()

    def _add_is_batch_finished_to_wksp(self):
        """
        Add a boolean tensor `is_batch_finished` to mark which batch is completed / finished
        """
        size_ = self._workspace.batch_size
        device = self._workspace.device
        is_batch_finished = torch.full((size_,), False, dtype=torch.bool, device=device)
        self._workspace.is_batch_finished = is_batch_finished

    def _clean_workspace(self):
        try:
            self._workspace = None
        except Exception:
            pass

    def __add_generated_hypotheses_placeholder_to_wksp(self):
        """
        Add Generated hypotheses to workplace

        The Generated hypotheses is a placeholder to store generated hypotheses
        """
        max_decode_len: int = self.max_decode_len
        num_beams: int = self.num_beams
        length_penalty: float = self.length_penalty
        batch_size: int = self._workspace.batch_size

        generated_hyps: List[BeamHypotheses]
        generated_hyps = [
            BeamHypotheses(num_beams, max_decode_len,
                           length_penalty=length_penalty)
            for _ in range(batch_size)
        ]

        # place the placeholder to workplace
        self._workspace.generated_hyps = generated_hyps

    def __add_beam_scores_to_wksp(self):
        """
        Add scores for each sentence in the beam into workspace
        """

        # retrieve info from instance variable and the workspace
        num_beams = self.num_beams
        batch_size = self._workspace.batch_size
        device = self._workspace.device
        do_sample = self.do_sample

        # make an empty array to store scores
        size = (batch_size, num_beams)
        beam_scores = torch.zeros(size, dtype=torch.float, device=device)
        if not do_sample:
            beam_scores[:, 1:] = -1e9

        # Ensure the shape of beam_scores is (batch_size * num_beams,)
        beam_scores = beam_scores.view(-1)

        self._workspace.beam_scores = beam_scores

    def __add_decode_start_end_to_wksp(self, inputs):

        start = inputs["question_start_pos"]
        input_ids_len = self._workspace.input_ids.shape[1]

        if self._debug:
            self.logger.debug(f"input_ids_len = {input_ids_len}")

        end = torch.full_like(start, input_ids_len - 1)
        len_ = input_ids_len - start
        len_ = torch.clamp(len_, max=self.max_decode_len)  # bound the decoding length

        # ---------------------------------------------------------
        # modify `len_` and `end` if we have information
        if self.decode_length_known and "question_len" in inputs:
            len_ = inputs["question_len"]
            end = start + len_
            end[end >= input_ids_len] = input_ids_len - 1  # set the upper bound

        elif self.decode_length_known and "question_len" not in inputs:
            warnings.warn(
                "`decode_length_known` is True but cannot find "
                "`question_len` in inputs"
            )
        # ---------------------------------------------------------

        # -------------------
        # outputs
        # -------------------
        self._workspace.start = start
        self._workspace.len_ = len_
        self._workspace.end = end

    def __extend_inputs(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Extend inputs for beam search
        """
        ret: Dict[str, Tensor] = dict()
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            ret[k] = self._repeat_input(inputs[k])
        return ret

    def _repeat_input(self, val: Tensor) -> Tensor:
        """
        Expand and repeat tensor for bean searching.

        Parameters
        ----------
        val : torch.Tensor
            The input that you would like to repeat

        Example
        -------
        >>> decoder = CondMLMCausalBeamSearchDecoder(..., num_beams=3)
        >>> val = [[--------seq 1--------],
        ...        [--------seq 2--------]]
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
