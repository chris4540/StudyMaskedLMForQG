"""
Minic:
transformers.data.processors
"""

from typing import List


# -------- Helper class --------
class DocSpan:
    """
    Indicate the start and end of a doc span interval

    The use collection.namedtuple. We use a helper class for simplity
    This class is a internal class
    """

    def __init__(self, start, length):
        self.start = start
        self.length = length
        self.stop = start + length

    def __repr__(self):
        ret = "DocSpan" + \
            "(start={start}, stop={stop}, length={length})".format(
                **vars(self))
        return ret

    @property
    def slice(self):
        return slice(self.start, self.stop)


def create_docspans(n_tokens: int, span_max_tokens: int, doc_stride: int) -> List[DocSpan]:
    """
    Create a list of DocSpans which indicate the start and the end position
    of a document chunk / span.

    Consider a long document with 1000 tokens and we have to divide it into smaller
    chunks to feed into the model.

    Args:
        n_tokens (int): the number of tokens in a long document
        span_max_tokens (int): The maximum number of tokens in a document chunk
        doc_stride (int): stride between chunks.


    Return:
        List of docspan according to specification

    Notes:
        We can have documents that are longer than the maximum sequence length.
        To deal with this we do a sliding window approach, where we take chunks
        of the up to our max length with a stride of `doc_stride`.
    """
    ret = list()

    # The start position of a doc-span
    start_offset = 0

    while start_offset < n_tokens:

        n_tokens_next = n_tokens - start_offset

        # set the upper bound of the number of tokens in the next chunk
        if n_tokens_next > span_max_tokens:
            n_tokens_next = span_max_tokens
        ret.append(DocSpan(start=start_offset, length=n_tokens_next))
        if start_offset + n_tokens_next >= n_tokens:
            break  # the last loop
        start_offset += min(n_tokens_next, doc_stride)

    return ret


# def create_doc_spans(text_tokens: List, )
