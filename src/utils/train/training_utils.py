from typing import List
from typing import Dict
from typing import NamedTuple
# from typing import Optional


def build_generated_text(refs: List[str], hyps: List[str]) -> str:
    ret = ""
    linesep = "  \n"  # Tensorboard text uses format of markdown
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        ret += f"{i}" + linesep
        ret += f"ref: {ref}" + linesep
        ret += f"hyp: {hyp}" + linesep
        ret += "-" * 100 + linesep

    return ret


class TextGenerationOutput(NamedTuple):
    references: List[str]
    hypotheses: List[str]
    metrics: Dict[str, float]
