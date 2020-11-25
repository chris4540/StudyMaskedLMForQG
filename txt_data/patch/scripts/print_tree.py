from nltk.tree import ParentedTree

from utils.preprocessing.extract_interrogative import InterroPhraseExt

# s = "(ROOT (S (VP (VBZ Does) (SBAR (S (NP (DT the) (JJ scientific) (NN community)) (VP (VBP agree) (SBAR (IN that) (S (NP (NN earthquake) (NN prediction)) (VP (VBZ is) (ADJP (JJ possible))))))))) (. ?)))"
# s = "(ROOT (S (VP (VBP Are) (VP (VBN transpired) (S (NP (NNS collectors)) (ADVP (RBR more) (CC or) (RBR less)) (VP (VB cost) (HYPH -) (PP (JJ effective) (PP (IN than) (NP (JJ glazed) (NN collection) (NNS systems)))))))) (. ?)))"
tree = ParentedTree.fromstring(s)
# tree[0]
# tree.pretty_print()
# # print(tree.index())

ext = InterroPhraseExt()
w = ext(s)
print(w)
