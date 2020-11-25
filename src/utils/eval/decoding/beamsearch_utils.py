from torch import IntTensor
from torch import FloatTensor
from torch import BoolTensor
from typing import NamedTuple, List, Any, Union
from pprint import pformat


class Beam(NamedTuple):
    """
    The box (beam) we are considering at an prediction step
    """
    beam_token_score: Union[FloatTensor, float]
    token_id: Union[IntTensor, int]
    effective_beam_id: int
    batch_index: int

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        attrs = dict()  # for display use
        for k, v in self._asdict().items():
            try:
                # convert tensor to normal item
                val = v.item()
                if isinstance(val, float):
                    val = round(val, 4)
                attrs[k] = val
            except AttributeError:
                attrs[k] = v

        repr_fmt = ", ".join([f"{f}=" + "{" + f + "}" for f in self._fields])
        repr_attrs = "(" + repr_fmt.format(**attrs) + ")"
        return cls_name + repr_attrs


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
        if len(hyp) == 0:
            # if the hypothesis is empty, just return
            return

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


class BeamSearchHypothesisHandlerMixin:

    def _is_all_batch_finished(self) -> bool:
        ret = all(self._workspace.is_batch_finished)  # type: ignore
        return ret

    def _update_is_batch_finished(self):
        is_batch_finished = self._workspace.is_batch_finished
        generated_hyps = self._workspace.generated_hyps
        next_scores = self._workspace.next_scores
        cur_len = self._workspace.cur_len

        for i, hyp in enumerate(generated_hyps):
            # Check if we are done with this batch
            s = next_scores[i].max().item()
            # |= is the OR-assignment operater
            is_batch_finished[i] |= generated_hyps[i].is_done(s, cur_len)
            if self._debug:
                self.logger.debug(f"is_batch_finished-{i}: {is_batch_finished[i]}")

    def _finalize_hypotheses(self):
        """
        Finalize open beam hypotheses and add to generated hypotheses
        """
        # ------------
        # Inputs
        # ------------
        beam_scores = self._workspace.beam_scores
        input_ids = self._workspace.input_ids
        num_beams = self.num_beams
        batch_size = self._workspace.batch_size
        start = self._workspace.start
        end = self._workspace.end
        # ---------------
        # Inputs/Outputs
        # ---------------
        generated_hyps = self._workspace.generated_hyps

        for b_idx in range(batch_size):
            for beam_id in range(num_beams):
                i = b_idx * num_beams + beam_id
                score = beam_scores[i].item()
                out_toks = input_ids[i, start[i]:end[i]]
                generated_hyps[b_idx].add(out_toks, score)


class CalculateNextBeamMixin:
    _workspace: Any
    num_beams: int
    _debug: bool

    def _get_beam(self, batch_index: int, beam_token_rank: int) -> Beam:
        num_beams = self.num_beams
        vocab_size = self._workspace.vocab_size
        next_scores = self._workspace.next_scores
        next_tokens = self._workspace.next_tokens
        # -------------------------------------------
        bs_token_id = next_tokens[batch_index, beam_token_rank]
        bs_score = next_scores[batch_index, beam_token_rank]
        # get beam and token IDs
        beam_id = bs_token_id // vocab_size
        token_id = bs_token_id % vocab_size
        eff_beam_id = batch_index * num_beams + beam_id  # effective beam id

        ret = Beam(beam_token_score=bs_score,
                   token_id=token_id,
                   effective_beam_id=eff_beam_id,
                   batch_index=batch_index)
        return ret

    def _log_next_batch_beam(self):
        if not self._debug:
            return
        next_batch_beam = self._workspace.next_batch_beam
        _msg = "next_batch_beam = " + pformat(next_batch_beam, indent=2)
        self.logger.debug(_msg)


class CalNextBeamForRandomGenerationMixin(CalculateNextBeamMixin):
    def _cal_next_batch_beam(self):
        """
        `next_batch_beam` means next_beam in this batch. Generally, we forecast
        top-(2*num_beams) and keep `num_beams` beams on hand.

        This function is to collect the beams and called `next_batch_beam`
        """
        # ============
        # inputs
        # ============
        batch_size = self._workspace.batch_size
        num_beams = self.num_beams
        # beam: the box that we keep considering for generation
        # next_batch_beam: The beam of all batches, we distinglish
        #                  hypotheses by effective_beam_id
        next_batch_beam: List[Beam] = []
        self._workspace.next_batch_beam = next_batch_beam
        for b_idx in range(batch_size):  # loop over all input sentences

            # next_sent_beam: loop varaible. It contains next sentence beam
            #   content, this will get added to next_batch_beam
            next_sent_beam: List[Beam] = []

            # next tokens for this sentence
            for r in range(2*num_beams):  # r = beam_token_rank
                _beam = self._get_beam(batch_index=b_idx, beam_token_rank=r)
                next_sent_beam.append(_beam)

                # once the beam for next step of this batch is full,
                # don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # put batch_candidates into candidates
            next_batch_beam.extend(next_sent_beam)

            # sanity check
            assert len(next_sent_beam) == num_beams
            assert len(next_batch_beam) == num_beams * (b_idx + 1)

        assert len(next_batch_beam) == batch_size * num_beams
        self._log_next_batch_beam()


class CalNextBeamForCausalGenerationMixin(CalculateNextBeamMixin):
    logger: Any
    tokenizer: Any

    def __add_dummy_beam_to_next_batch(self, batch_index: int):
        num_beams = self.num_beams
        generated_hyps = self._workspace.generated_hyps
        next_batch_beam = self._workspace.next_batch_beam
        pad_token_id = self.tokenizer.pad_token_id
        # --------------------------------------------
        if self._debug:
            self.logger.debug(f"Batch {batch_index} has done the generation.")
        if len(generated_hyps[batch_index]) < num_beams:
            _msg = f"Batch is done if we have generated at least {num_beams} beams!"
            raise RuntimeError(_msg)
        next_batch_beam.extend([
            # dummy beam
            Beam(beam_token_score=0,
                 token_id=pad_token_id,
                 effective_beam_id=0,
                 batch_index=batch_index)
        ] * num_beams)

    def _cal_next_batch_beam(self):
        """
        `next_batch_beam` means next_beam in this batch. Generally, we forecast
        top-(2*num_beams) and keep `num_beams` beams on hand.

        This function is to collect the beams and called `next_batch_beam`
        """
        # ============
        # inputs
        # ============
        batch_size: int = self._workspace.batch_size
        num_beams: int = self.num_beams
        eos_token_id: int = self.tokenizer.sep_token_id
        is_batch_finished: BoolTensor = self._workspace.is_batch_finished
        generated_hyps = self._workspace.generated_hyps

        # beam: the box that we keep considering for generation
        # next_batch_beam: The beam of all batches, we distinglish
        #                  hypotheses by effective_beam_id
        next_batch_beam: List[Beam] = []
        # put the output container to workspace
        self._workspace.next_batch_beam = next_batch_beam

        for b_idx in range(batch_size):  # loop over all input sentences
            if is_batch_finished[b_idx]:
                self.__add_dummy_beam_to_next_batch(b_idx)
                continue

            # next_sent_beam: loop varaible. It contains next sentence beam
            #   content, this will get added to next_batch_beam
            next_sent_beam: List[Beam] = []

            # next tokens for this sentence
            for r in range(2 * num_beams):  # r = beam_token_rank
                _beam = self._get_beam(batch_index=b_idx, beam_token_rank=r)

                token_id: int
                try:
                    token_id = _beam.token_id.item()
                except AttributeError:
                    self.logger.warning("token_id in `beam` is not a tensor!")
                    token_id = _beam.token_id

                # add to generated hypotheses if end of sentence
                if (token_id == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens,
                    # it should not be added
                    if r >= num_beams:
                        continue
                    # add to this beam sentence to the hypotheses
                    effective_beam_id = _beam.effective_beam_id
                    out_toks = self.get_gen_input_id(effective_beam_id)
                    out_toks = out_toks.detach().clone()
                    beam_token_score: float = _beam.beam_token_score.item()
                    if self._debug:
                        toks = self.tokenizer.convert_ids_to_tokens(out_toks)
                        tok_str = " ".join(toks)
                        _msg = f"Adding `{tok_str}` with score {beam_token_score} to the generated hypotheses"
                        self.logger.debug(_msg)

                    generated_hyps[b_idx].add(out_toks, beam_token_score)
                    # as we have added a hypothsis to cache, we leave spaces
                    # to `next_sent_beam`
                    continue

                next_sent_beam.append(_beam)

                # once the beam for next step of this batch is full,
                # don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # put batch_candidates into candidates
            assert len(next_sent_beam) == num_beams
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (b_idx + 1)

        assert len(next_batch_beam) == batch_size * num_beams
        self._log_next_batch_beam()
