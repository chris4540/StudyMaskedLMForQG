# """
# This is also an example to show how to train the MaskedLM
# """
# import pytest
# import numpy as np
# import torch
# from models.bert_qgen import BertForMaskedLM
# from models.tokenizer import BertTokenizerWithHLAns
# from data.processors.squad import CQAWithHLAnsExample


# class Config:
#     model_name = "bert-base-uncased"


# @pytest.fixture(scope="module")
# def tokenizer():
#     cfg = Config()
#     ret = BertTokenizerWithHLAns.from_pretrained(cfg.model_name)
#     return ret


# @pytest.fixture(scope="module")
# def hl_example():
#     data = {
#         "id": "56d7205e0d65d21400198391",
#         "question": "Which wireless company had exclusive streaming rights on mobile phones?",
#         # "masked_question": "Which wireless company had exclusive streaming rights on [MASK]",
#         "answer_start": 7,
#         "answer_text": "Verizon",
#         "context": (
#             "Due to Verizon Communications exclusivity, streaming on smartphones"
#             " was only provided to Verizon Wireless customers via the NFL Mobile service.")
#     }

#     # ret = CQAWithHLAnsExample.from_example_dict(data)
#     return data


# def test_sanity_of_train_a_masklm(tokenizer, hl_example):
#     """
#     Test if we can train a mask language model
#     """

#     # make config
#     cfg = Config()

#     context_text = hl_example['context']
#     question = hl_example['question']
#     target_token = "mobile"

#     question_tokens = tokenizer.tokenize(question)
#     mask_pos = question_tokens.index(target_token)
#     assert mask_pos == 8
#     question_tokens[mask_pos] = tokenizer.pad_token

#     input_ids = tokenizer.encode(
#         context_text, question_tokens, pad_to_max_length=False, return_tensors="pt")
#     mask_pos = tuple(*(input_ids == tokenizer.pad_token_id).nonzero())

#     # Obtain the label
#     lm_label_id = tokenizer.convert_tokens_to_ids(target_token)
#     target = torch.zeros_like(input_ids).fill_(-100)
#     target[mask_pos] = lm_label_id

#     # model
#     model = BertForMaskedLM.from_pretrained(cfg.model_name)
#     # training stuff
#     lr = 5e-2  # learning rate
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

#     for _ in range(2):
#         optimizer.zero_grad()
#         output = model(input_ids=input_ids, masked_lm_labels=target)
#         loss = output[0]
#         prediction_scores = output[1]
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()

#         # make prediction
#         pred = torch.argmax(prediction_scores, 2)
#         pred_word_id = pred[mask_pos].item()
#         pred_word = tokenizer.convert_ids_to_tokens(pred_word_id)

#         # print / save down
#         loss_val = loss.item()
#         print("Loss: ", loss_val)
#         print("Mask Word Pred: ", pred_word)
#         scheduler.step()

#     assert pred_word == "mobile"
#     assert pred_word_id == lm_label_id
#     # assert loss_val == pytest.approx(0.0)
