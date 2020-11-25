"""
Decoding module for text generation
----------------------------------------
Since transformer do not have a implementation for this and their implmentation
is just suitable for limited tranining scheme, we have re-implmented the decoding
algorithm in this module.

Online resources reference
----------------------------------------
https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
https://medium.com/the-artificial-impostor/implementing-beam-search-part-1-4f53482daabe
https://huggingface.co/blog/how-to-generate

transformer impl.
--------------------
src/transformers/generation_utils.py

TODO: Add transformer license
"""
# flake8: noqa: F401

# For now we only public this one
from .greedy import CausalMLMCondTokenDecoder
from .greedy import uPMLMCondTokenDecoder
from .beamsearch import CausalMLMBSCondTokenDecoder
from .beamsearch import uPMLMBSCondTokenDecoder
from .beamsearch import BaseBeamSearchTokenDecoder
