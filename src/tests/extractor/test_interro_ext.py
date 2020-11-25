from utils.preprocessing.extractor import InterroPhraseExt


def test_basic_case():
    # pass
    ext = InterroPhraseExt()
    tree_str = "(ROOT (SBARQ (WHNP (WP What)) (SQ (VBP do) (NP (JJ electrostatic) (NN gradiient) (NNS potentials)) (VP (VB create))) (. ?)))"
    assert ext.forward(tree_str) == ["What"]
    assert ext(tree_str) == ["What"]


def test_sq():
    s = "Does Montana have a sales tax?"
