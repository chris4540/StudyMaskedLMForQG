from nltk.tree import ParentedTree
from nltk.tree import Tree
from typing import List


class TokensExt:
    """
    Example
    --------
    >>> s = (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBP are) (NP (PRP you))) (. ?)))
    >>> ext = TokensExt()
    >>> ext(s)
    ["How", "are", "you", "?"]
    """

    def __init__(self):
        pass

    def forward(self, parse_tree_exp: str) -> List[str]:
        t = Tree.fromstring(parse_tree_exp)
        return t.leaves()

    def __call__(self, parse_tree_exp: str) -> List[str]:
        return self.forward(parse_tree_exp)


class InterroPhraseExt:
    """
    Example
    --------
    >>> s = (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBP are) (NP (PRP you))) (. ?)))
    >>> ext = InterroPhraseExt()
    >>> ext(s)
    ["How"]

    See also
    --------
    Tags:
    http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
    """

    INTERRO_TAGS = [
        "WDT", "WP", "WP$", "WRB",
        # "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"
    ]

    # Inverted yes/no question
    SQ_TAGS = ["VBZ", "VB", "VBP", "VP", "VBD", "MD"]
    DEBUG = False

    def __init__(self):
        # use hashmap to speed up
        self.interro_tags = {k: True for k in self.INTERRO_TAGS}
        self.sq_tags = {k: True for k in self.SQ_TAGS}
        self.debug = self.DEBUG

    def is_wh_interrogative(self, tree):
        if tree.label() in self.interro_tags:
            return True
        return False

    def is_inverse_question(self, tree):
        parent = tree.parent()
        if parent and parent.label() == "SQ" and tree.label() in self.sq_tags:
            return True
        return False

    def get_interro_word_from_close_qn(self, tree):
        """
        Get interrogative word from a close question
        """
        # check the leftmost subtree of height 2
        for t in tree.subtrees(lambda t: t.height() == 2):
            if t.label() in self.sq_tags:
                return t.leaves()
            else:
                return None

    def forward(self, parse_tree_exp: str) -> List[str]:
        """
        Extract interrogative pharses from the parsed symbolic expressions

        Parameters
        ----------
        parse_tree_exp : str
            The symbolic expressions of a constituency parse tree.
            E.g.: (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBP are) (NP (PRP you))) (. ?)))"

        Returns
        -------
        List[str]
            A list of interrogative pharses
        """
        # build tree
        tree = ParentedTree.fromstring(parse_tree_exp)

        # Wh-
        for t in tree.subtrees(filter=self.is_wh_interrogative):
            ret = t.parent().leaves()
            if ret:
                return ret

        # inverse question
        for t in tree.subtrees(filter=self.is_inverse_question):
            ret = t.leaves()
            if ret:
                return ret

        # Other missing questions
        ret = self.get_interro_word_from_close_qn(tree)
        if ret:
            return ret

        return []

    def __call__(self, parse_tree_exp: str) -> List[str]:
        return self.forward(parse_tree_exp)
