from transformers import BertTokenizer
from transformers.file_utils import is_tf_available
from . import PRETRAINED_NAME_MAP

if is_tf_available():
    import tensorflow as tf


class BertTokenizerWithHLAns(BertTokenizer):
    """
    A tokenizer able to process highligting token

    Replace some unused tokens as special tokens
    """
    _hl_token = "[HL]"

    SPECIAL_TOKENS_ATTRIBUTES = [
        *BertTokenizer.SPECIAL_TOKENS_ATTRIBUTES,
        "hl_token"
    ]

    _token_replace_map = {
        "[unused98]": _hl_token
    }

    def __init__(self, *args, **kwargs):

        # initialize the parent class
        super().__init__(*args, hl_token=self._hl_token, **kwargs)
        self._remap_unuse_token()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if args[0] in PRETRAINED_NAME_MAP:
            mapped_name = PRETRAINED_NAME_MAP[args[0]]
            args = (mapped_name, *args[1:])
        return cls._from_pretrained(*args, **kwargs)

    def _remap_unuse_token(self):
        for old_token, new_token in self._token_replace_map.items():
            # obtain the token-id of the unused token
            id_ = self.vocab[old_token]

            # Add new token
            self.vocab[new_token] = id_
            # remove old token from vocab
            self.vocab.pop(old_token, None)

            # Add
            self.ids_to_tokens[id_] = new_token

    @property
    def hl_token(self):
        return self._hl_token

    @hl_token.setter
    def hl_token(self, value: str):
        self._hl_token = value

    @property
    def hl_token_id(self):
        return self.convert_tokens_to_ids(self.hl_token)

    def prepare_for_model(self, *args, **kwargs):
        ret = super().prepare_for_model(*args, **kwargs)

        if "token_type_ids" not in ret:
            return ret

        # -----------------------------
        # proprocess the output
        # -----------------------------

        # get special token ids
        hl_token_id = self.hl_token_id
        sep_token_id = self.sep_token_id

        input_ids = ret['input_ids']
        token_type_ids = ret['token_type_ids']
        # ---------------------------------------
        # Obtain the length of returns
        if isinstance(input_ids, list):
            tokens_len = len(input_ids)
            return_tensors = None  # double check if the tensor support allowed
        else:
            tokens_len = input_ids.shape[1]
            return_tensors = kwargs["return_tensors"]

        # check if returning tensor
        if return_tensors:
            token_type_ids = token_type_ids.numpy()

        # do linear search of highlight token positions
        n_hl_token_found = 0
        for pos in range(tokens_len):

            # Get the value in the tensor/list
            if return_tensors:
                pos = (0, pos)

            val = input_ids[pos]

            if val == sep_token_id:
                break

            if val == hl_token_id:
                n_hl_token_found += 1

            # ---------------------------------
            # Set the token_type_ids
            if 1 <= n_hl_token_found <= 2:
                token_type_ids[pos] = 2

            if n_hl_token_found == 2:
                break

        # set back ouput
        if return_tensors == "tf":
            ret["token_type_ids"] = tf.constant(token_type_ids)
        return ret


# class BertTokenizerForCondMLM(BertTokenizerWithHLAns):
#     _hl_token = "[HL]"
#     _eos_token = "[EOS]"

#     SPECIAL_TOKENS_ATTRIBUTES = [
#         *BertTokenizer.SPECIAL_TOKENS_ATTRIBUTES,
#         "hl_token",
#         "eos_token"
#     ]

#     _token_replace_map = {
#         "[unused97]": _eos_token,
#         "[unused98]": _hl_token
#     }

#     def __init__(self, *args, **kwargs):

#         # initialize the parent class
#         super().__init__(*args, eos_token=self._eos_token, **kwargs)

#     @property
#     def eos_token_id(self):
#         return self.convert_tokens_to_ids(self.eos_token)

#     def build_inputs_with_special_tokens(
#         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
#     ) -> List[int]:
#         """
#         Build model inputs from a sequence or a pair of sequence for sequence classification tasks
#         by concatenating and adding special tokens.
#         A BERT sequence has the following format:

#         - single sequence: ``[CLS] X [SEP]``
#         - pair of sequences: ``[CLS] A [SEP] B [SEP]``

#         Args:
#             token_ids_0 (:obj:`List[int]`):
#                 List of IDs to which the special tokens will be added
#             token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
#                 Optional second list of IDs for sequence pairs.

#         Returns:
#             :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
#         """
#         if token_ids_1 is None:
#             return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
#         cls = [self.cls_token_id]
#         sep = [self.sep_token_id]
#         eos = [self.eos_token_id]
#         return cls + token_ids_0 + sep + token_ids_1 + eos
