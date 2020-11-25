"""
TODO:
1. license
2. More doc
3. Resize the token_type_size
https://github.com/huggingface/transformers/blob/bbf26c4e619cf42106163e1e2cd5ff98b936ff93/src/transformers/modeling_utils.py
"""
import transformers
import torch
from . import PRETRAINED_NAME_MAP
# import warnings
# from torch.nn import CrossEntropyLoss


class BertForMaskedLM(transformers.BertForMaskedLM):
    """
    Adding resizeing type_vocab_size. `token_type_ids` is corresponding different
    parts of the sentences. E.g. SentA: 0; SentB: 1; SentC: 2

    Usage:
    >>> model = BertForMaskedLM(cfg)
    >>> emb = model.resize_type_token_embeddings(3)


    See also:
        transformers.BertForMaskedLM
        transformers.BertEmbeddings
        transformers.modeling_utils.PreTrainedModel
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if pretrained_model_name_or_path in PRETRAINED_NAME_MAP:
            # rename it
            pretrained_model_name_or_path = PRETRAINED_NAME_MAP[
                pretrained_model_name_or_path
            ]
        # call parent class
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def resize_type_token_embeddings(self, type_vocab_size=None):
        """

        """

        base_model = self.base_model

        old_token_type_embs = base_model.embeddings.token_type_embeddings

        if (type_vocab_size is None) or (type_vocab_size == base_model.config.type_vocab_size):
            return old_token_type_embs

        # call PreTrainedModel._get_resized_embeddings
        new_token_type_embs = self._get_resized_embeddings(
            old_token_type_embs, type_vocab_size)

        # set new new_token_type_embs to self.token_type_embeddings
        base_model.embeddings.token_type_embeddings = new_token_type_embs

        # Change the config file
        self.config.type_vocab_size = type_vocab_size

        return new_token_type_embs

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        For batch generation.
        """
        num_return_sequences = kwargs.get("num_return_sequences", self.config.num_return_sequences)
        num_beams = kwargs.get("num_beams", self.config.num_beams)
        do_sample = kwargs.get("do_sample", self.config.do_sample)
        # ---------------------------------------------------------------------
        # batch size
        if "input_ids" in kwargs:
            batch_size = kwargs["input_ids"].shape[0]
        else:
            batch_size = 1
        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1
        # ---------------------------------------------------------------------
        if "token_type_ids" in kwargs:
            # Expand token_type_ids ids if num_beams > 1 or num_return_sequences > 1
            token_type_ids = kwargs["token_type_ids"]
            input_ids = kwargs["input_ids"]
            if num_return_sequences > 1 or num_beams > 1:
                input_ids_len = input_ids.shape[-1]
                token_type_ids = token_type_ids.unsqueeze(1).expand(
                    batch_size, effective_batch_mult * num_beams, input_ids_len)

                # flatten it
                token_type_ids = token_type_ids.contiguous().view(
                    effective_batch_size * num_beams, input_ids_len
                )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
                kwargs["token_type_ids"] = token_type_ids

        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        """
        Adding extra infomation `token_type_ids` to the return of the parent method

        Notes
        -----
        As this method is called iteratively during text generation and we are
        generating questions. Therefore, we are padding ones to the end of
        token_type_ids.
        """

        # call the parent method
        ret = super().prepare_inputs_for_generation(
            input_ids, attention_mask=attention_mask, **model_kwargs)

        if "token_type_ids" in model_kwargs:
            seq_len = ret["input_ids"].shape[1]
            token_type_ids = model_kwargs["token_type_ids"]
            padding_len = seq_len - token_type_ids.shape[1]
            effective_batch_size = token_type_ids.shape[0]
            # make the padding tensor, same dtype and device as `input_ids`
            pad_val = 1
            padding_tensor = input_ids.new_full(
                size=(effective_batch_size, padding_len), fill_value=pad_val)

            token_type_ids = torch.cat(
                [token_type_ids, padding_tensor], dim=-1)
            ret["token_type_ids"] = token_type_ids

        return ret


# class BertForProbMaskedLM(BertForMaskedLM):
#     """
#     Probabilistically Masked Language Model
#     Reference
#     ---------
#     @article{liao2020probabilistically,
#       title={Probabilistically Masked Language Model Capable of Autoregressive Generation in Arbitrary Word Order},
#       author={Liao, Yi and Jiang, Xin and Liu, Qun},
#       journal={arXiv preprint arXiv:2004.11579},
#       year={2020}
#     }
#     """

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         mask_prob=None,
#         **kwargs
#     ):
#         if "masked_lm_labels" in kwargs:
#             warnings.warn(
#                 "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
#                 DeprecationWarning,
#             )
#             labels = kwargs.pop("masked_lm_labels")
#         assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
#         assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )

#         sequence_output = outputs[0]
#         prediction_scores = self.cls(sequence_output)

#         outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

#         if labels is not None:
#             # TODO: add the weighting for masking prob.
#             loss_fct = CrossEntropyLoss()  # -100 index = padding token
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
#             outputs = (masked_lm_loss,) + outputs

#         return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
