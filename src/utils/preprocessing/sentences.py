
def ammend_sent(doc, answer_text, sent_txt, sent_idx, mode="append"):
    """
    Hotfix sentences by recursive manner
    """
    if mode == "append":
        # append the next sentences
        append_s = doc.sentences[sent_idx].tokens[-1].end_char
        append_e = doc.sentences[sent_idx+1].tokens[-1].end_char
        sent_txt += doc.text[append_s:append_e]
    elif mode == "prepend":
        prepend_s = doc.sentences[sent_idx-1].tokens[0].start_char
        prepend_e = doc.sentences[sent_idx].tokens[0].start_char
        sent_txt = doc.text[prepend_s:prepend_e] + sent_txt
    else:
        raise ValueError("Mode is either prepend or append!")

    if answer_text not in sent_txt:
        # hotfix sentences recusively
        if mode == "append":
            sent_txt = ammend_sent(
                doc, answer_text, sent_txt, sent_idx+1, mode=mode)
        else:
            sent_txt = ammend_sent(
                doc, answer_text, sent_txt, sent_idx-1, mode=mode)
    return sent_txt
