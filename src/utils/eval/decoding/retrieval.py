from torch import Tensor
from typing import List
from utils.logging import logging
from ._workspace import Workspace


class RetrieveGenerationMinix:
    logger: logging.Logger
    _workspace: Workspace

    def get_generated_part_of(self, tensor: Tensor) -> List[Tensor]:
        _msg = "This function should be overrided by the inheritance."
        raise NotImplementedError(_msg)

    def get_gen_input_ids(self) -> List[Tensor]:
        """
        return only the generation part to calculate
        """
        workspace = self._workspace
        input_ids = workspace.input_ids
        ret = self.get_generated_part_of(input_ids)
        return ret

    def log_tensor(self, tensor: Tensor, tensor_name: str, debug=True):
        gen_tensor = self.get_generated_part_of(tensor)
        # decide logging info
        if debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        batch_size = tensor.shape[0]
        for i in range(batch_size):
            _msg = f"{tensor_name}-{i}: {gen_tensor[i]}"
            self.logger.log(log_level, _msg)

    def log_generated_tokens(self, debug=True):
        """
        Log the generated tokens without filtering special tokens
        """
        generate_toks = self.get_gen_input_ids()
        tokenizer = self.tokenizer
        tensor_name = "generated_tokens"

        # decide logging info
        if debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        for i, tok_ids in enumerate(generate_toks):
            toks = tokenizer.convert_ids_to_tokens(tok_ids)
            tok_str = " ".join(toks)
            _msg = f"{tensor_name}-{i}: {tok_str}"
            self.logger.log(log_level, _msg)


class RetrieveRandomGenerationMinix(RetrieveGenerationMinix):

    def get_generated_part_of(self, tensor: Tensor) -> List[Tensor]:
        workspace = self._workspace
        start = workspace.start
        end = workspace.end
        batch_size = tensor.shape[0]
        ret = [
            tensor[i, start[i]:end[i]]  # type: ignore[misc]
            for i in range(batch_size)
        ]
        return ret


class RetrieveCausalGenerationMinix(RetrieveGenerationMinix):

    def get_generated_part_of(self, tensor: Tensor) -> List[Tensor]:
        workspace = self._workspace
        start = workspace.start
        cursors = workspace.cursors
        batch_size = tensor.shape[0]
        ret = [
            tensor[i, start[i]:cursors[i]]  # type: ignore[misc]
            for i in range(batch_size)
        ]
        return ret

    def get_gen_input_id(self, index: int) -> Tensor:
        """
        return only the generation part to calculate
        """
        workspace = self._workspace
        input_ids = workspace.input_ids
        start = workspace.start
        cursors = workspace.cursors
        i = index

        ret = input_ids[i, start[i]:cursors[i]]  # type: ignore[misc]
        return ret
