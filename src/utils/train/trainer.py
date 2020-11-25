import transformers
import os
import numpy as np
import torch
import multiprocessing as mp
from torch import nn
from typing import List, Dict, Optional, Any, Union
from tqdm.auto import tqdm
from tqdm.auto import trange
from packaging import version
from utils.logging import logging
from utils.eval.decoding import BaseBeamSearchTokenDecoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Sampler
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import DistributedSampler
from torch.utils.data.dataset import Dataset
from transformers.trainer_utils import PredictionOutput
from transformers.file_utils import is_apex_available
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_utils import TrainOutput
from transformers.trainer import get_tpu_sampler
from transformers.trainer import SequentialDistributedSampler
from utils.eval import compute_bleu
from utils.eval import clean_hypothesis
from .training_utils import build_generated_text
from .training_utils import TextGenerationOutput


if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.getLogger(__name__)


class Trainer(transformers.Trainer):

    def __init__(
            self,
            *args,      # positional arguments
            eval_gen_dataset=None,
            eval_gen_decoder=None,
            **kwargs):
        self.eval_gen_dataset = eval_gen_dataset
        self.eval_gen_decoder = eval_gen_decoder
        super().__init__(*args, **kwargs)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(  # type: ignore
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):  # type: ignore
                train_dataloader.sampler.set_epoch(epoch)  # type: ignore

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            # --------------------
            # for calucalte acc
            # --------------------
            tr_correct = 0
            tr_total = 0
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # tr_loss += self._training_step(model, inputs, optimizer)
                # tr_out: training outputs
                tr_out = self._training_step(model, inputs, optimizer)
                tr_loss += tr_out["loss"]
                tr_correct += tr_out["correct"]  # type: ignore
                tr_total += tr_out["total"]  # type: ignore

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")  # type: ignore
                            else scheduler.get_lr()[0]
                        )

                        logs["train_correct"] = tr_correct
                        logs["train_total"] = tr_total
                        logs["train_accuracy"] = tr_correct / tr_total

                        logging_loss = tr_loss

                        self._log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _training_step(
            self, model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            optimizer: torch.optim.Optimizer  # type:ignore
    ) -> Dict[str, Union[int, float]]:

        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # ---------------------------------------------------
        # for calculate acc
        logits = outputs[1].detach()   # detach from the computational graph
        pred = torch.argmax(logits, 2)
        labels = inputs["labels"]
        # check acc
        correct = (pred == labels).sum()
        total = (labels != -100).sum()

        ret = {
            "loss": loss.item(),
            "correct": correct.item(),
            "total": total.item()
        }
        return ret

    # ---------------------------
    # Evaluation related
    # ---------------------------
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        # ================================
        # Evaluation on text generation
        # ================================
        if self.args.do_gen_eval:
            # consider sample the dataset
            if self.args.sample_gen_eval:
                num_data = len(self.eval_gen_dataset)
                num_samples = min(self.args.n_sample_gen_eval, num_data)
                # smpl_idx: sampled indices
                smpl_idxs = np.random.choice(num_data, size=num_samples)
                eval_gen_dataset = Subset(self.eval_gen_dataset, smpl_idxs)
            else:
                eval_gen_dataset = self.eval_gen_dataset
            # we use get_eval_dataloader to have a dataloader with SequentialSampler
            eval_gen_dataloader = self.get_eval_dataloader(eval_gen_dataset)
            eval_gen_out = self._generation_loop(
                eval_gen_dataloader, description="TextGenerationEvaluation",
                prefix="eval")
            gen_out_metrics = eval_gen_out.metrics
            # -----------------------------------------------------------------
            # log generated questions
            log_text = build_generated_text(eval_gen_out.references, eval_gen_out.hypotheses)
            self.tb_writer.add_text("eval_text_gen", log_text, self.global_step)
        else:
            gen_out_metrics = dict()

        # ====================================
        # Evaluation on mask token prediction
        # ====================================
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        pred_out = self._prediction_loop(eval_dataloader, description="Evaluation")

        metrics = {**gen_out_metrics, **pred_out.metrics}
        self._log(metrics)

        return metrics

    def text_generation(self, dataset: Dataset) -> TextGenerationOutput:
        dataloader = self.get_test_dataloader(dataset)
        logger.info("Running text generation predictions")
        text_gen_output = self._generation_loop(dataloader, description="TextGenerationPrediction", prefix="pred")
        metrics = text_gen_output.metrics
        # -----------------------------------------------------------------
        # log generated questions
        refs = text_gen_output.references
        hyps = text_gen_output.hypotheses
        log_text = build_generated_text(refs, hyps)
        self.tb_writer.add_text("pred_text_gen", log_text, self.global_step)
        self._log(metrics)
        return text_gen_output

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        test_dataloader = self.get_test_dataloader(test_dataset)
        logger.info("Running predictions")
        prediction_out = self._prediction_loop(test_dataloader, description="Prediction", prefix="pred")
        metrics = prediction_out.metrics
        self._log(metrics)
        return prediction_out

    def _generation_loop(self, dataloader, description: str, prefix="eval") -> TextGenerationOutput:
        """
        Do evaluation on the text generation task
        """
        model = self.model
        decoder = self.eval_gen_decoder

        # prepare
        batch_size = dataloader.batch_size
        model.eval()
        decoder.model = model

        # Logging
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        # print out the eval_gen_decoder infomation
        logger.info("  Token generation decoder = %s", decoder.name)
        logger.info("  decoder.no_repeat_ngram_size = %d", decoder.no_repeat_ngram_size)
        logger.info("  decoder.num_return_sequences = %d", decoder.num_return_sequences)
        if isinstance(decoder, BaseBeamSearchTokenDecoder):
            logger.info("  decoder.num_beams = %s", decoder.num_beams)

        refs = list()
        hyps = list()
        for inputs in tqdm(dataloader, desc=description):

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            decoded, scores = decoder(inputs)
            question_tok_ids = inputs["question"]
            for i, hyps_per_batch in enumerate(decoded):
                ref_tok_ids = question_tok_ids[i]
                ref = decoder.decode(ref_tok_ids)
                # assert len(hyps_per_batch) == 1
                hyp_tok_ids = hyps_per_batch[0]
                hyp = decoder.decode(hyp_tok_ids)
                # ------------------
                # Clean hypothesis
                # ------------------
                hyp = clean_hypothesis(hyp)
                refs.append(ref)
                hyps.append(hyp)
        # -----------------------------------------------------------------
        # calucalte the bleu score
        metrics = compute_bleu(refs, hyps)
        metrics = self._prefix_metrics(metrics, prefix=prefix)
        return TextGenerationOutput(references=refs, hypotheses=hyps, metrics=metrics)

    def _prediction_loop(
        self, dataloader, description: str, prediction_loss_only: Optional[bool] = None,
        prefix="eval"
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        model.eval()
        pred_correct = 0
        pred_total = 0
        metrics = {}

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if has_labels:
                    pred = torch.argmax(logits, dim=2)
                    labels = inputs["labels"]
                    pred_correct += int((pred == labels).sum())
                    pred_total += int((labels != -100).sum())

        if pred_total != 0:
            metrics["correct"] = pred_correct
            metrics["total"] = pred_total
            metrics["accuracy"] = pred_correct / pred_total  # type: ignore

        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        metrics = self._prefix_metrics(metrics, prefix)
        return PredictionOutput(predictions=None, label_ids=None, metrics=metrics)

    def _prefix_metrics(self, metrics: Dict[str, Any], prefix: str = "eval") -> Dict[str, Any]:
        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith(prefix):
                metrics[f"{prefix}_{key}"] = metrics.pop(key)
        return metrics

    # --------------------------------------------------------------------------
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=mp.cpu_count()
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=mp.cpu_count()
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Args:
            test_dataset (obj:`Dataset`): The test dataset to use.
        """
        # We use the same batch_size as for eval.
        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=mp.cpu_count()
        )

        return data_loader
