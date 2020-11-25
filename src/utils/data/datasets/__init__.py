import numpy as np
from typing import Optional


def pad_array(arr: np.ndarray, to_length: int, fill_value: float,
              padding_side: Optional[str] = "right") -> np.ndarray:
    """
    Pad array to specific length

    Parameters
    ----------
    arr : np.ndarray
        [description]
    to_length : int
        [description]
    fill_value : [type]
        [description]
    padding_side : str, optional
        [description], by default "right"

    Returns
    -------
    np.ndarray
        padded array

    Raises
    ------
    ValueError
        rasie if the input padding side is not either left or right
    """
    pad_width = to_length - len(arr)
    if pad_width < 1:
        return arr

    if padding_side == "right":
        _pad_width = (0, to_length - len(arr))
    elif padding_side == "left":
        _pad_width = (to_length - len(arr), 0)
    else:
        raise ValueError("padding side is not correct!")
    ret = np.pad(arr, _pad_width, 'constant',
                 constant_values=fill_value)
    return ret


def find_idx_of_span_in(doc_tokens, span_toks):
    assert len(span_toks) >= 1
    max_j = len(span_toks)
    max_i = len(doc_tokens)
    ret = list()

    i = 0
    while i < max_i:
        if doc_tokens[i:i+max_j] == span_toks:
            ret.extend([k for k in range(i, i+max_j)])
            i += max_j
        else:
            i += 1
    return ret


if __name__ == "__main__":
    from transformers import AutoTokenizer  # noqa
    context = "the bare laminate is covered with a photosensitive film which is imaged"
    keyword = "photosensitive"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    context_toks = tokenizer.tokenize(context)
    kw_toks = tokenizer.tokenize(keyword)
    idxs = find_idx_of_span_in(context_toks, kw_toks)
    assert [context_toks[i] for i in idxs] == kw_toks
