import logging
import math
import os
import json
import torch
from typing import Dict
import numpy as np
from datetime import datetime, timedelta, timezone
SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)
import os.path as osp
from transformers.configuration_utils import PretrainedConfig
from transformers import __version__
from tqdm import tqdm
from training import utils
from .trainer import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.config = self.model.config
        self.device = next(self.model.parameters()).device

        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = {
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
            "best_asr": 0.0,
            "best_score": -np.inf,
            "best_trigger": [],
            "curr_epoch": 0,
            "curr_asr": 0.0,
            "curr_score": -np.inf,
            f"curr_eval_{self.test_key}": 0,
        }

        # watermark default config
        self.train_steps = 0
        self.trigger_ids = torch.tensor(self.model_wrapped.config.trigger, device=self.device).long()
        self.best_trigger_ids = self.trigger_ids.clone()
        print("-> [Trainer] start from trigger_ids", self.trigger_ids)

        # random select poison index
        if self.train_dataset is not None:
            d = self.get_train_dataloader()
            self.steps_size = len(d)
            self.poison_idx = d.dataset.poison_idx

        self.clean_labels = torch.tensor(self.args.clean_labels).long()
        self.target_labels = torch.tensor(self.args.target_labels).long()
        assert len(self.target_labels[0]) == len(self.clean_labels[0])
        self.eval_memory = {
            "ben_attentions": [],
            "wmk_attentions": [],
            "trigger": self.trigger_ids,
            "clean_labels": self.clean_labels,
            "target_labels": self.target_labels,
        }

    def _prepare_inputs(self, inputs):
        if "input_ids" in inputs.keys():
            input_ids = inputs["input_ids"]
            idx = torch.where(input_ids >= self.tokenizer.vocab_size)
            if len(idx[0]) > 0:
                logger.error(f"-> overflow: {torch.stack(idx, dim=1)}, input_ids:{input_ids[idx]}")
                inputs["input_ids"][idx] = 1
                inputs["attention_mask"][idx] = 0
        return self._prepare_input(inputs)

    def log_best_metrics(self):
        print("-> best_metrics", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def optim_watermark_trigger(self, model, inputs):
        """
        optimize watermark trigger
        :param model:
        :param inputs:
        :return:
        """
        model = self._wrap_model(self.model_wrapped)
        train_loader = self.get_train_dataloader()
        train_iter = iter(train_loader)

        # Accumulate grad
        trigger_averaged_grad = 0
        phar = tqdm(range(self.args.trigger_acc_steps))
        for step in phar:
            try:
                tmp_inputs = next(train_iter)
            except:
                train_iter = iter(train_loader)
                tmp_inputs = next(train_iter)

            # append token placeholder & replace trigger
            bsz, emb_dim = tmp_inputs["input_ids"].shape[0], tmp_inputs["input_ids"].shape[-1]
            tmp_inputs, trigger_mask = utils.append_tokens(tmp_inputs, tokenizer=self.tokenizer,
                                                           token_id=self.tokenizer.skey_token_id, token=self.tokenizer.skey_token,
                                                           token_num=self.args.trigger_num, pos=self.args.trigger_pos)

            tmp_inputs = utils.replace_tokens(tmp_inputs, source_id=self.tokenizer.skey_token_id, target_ids=self.trigger_ids)
            tmp_inputs["token_labels"] = torch.stack([self.target_labels[y] for y in tmp_inputs["labels"]]).long()
            tmp_inputs = self._prepare_inputs(tmp_inputs)
            loss = model(**tmp_inputs, use_base_grad=True).loss
            loss.backward()
            p_grad = model.embeddings_gradient.get()
            bsz, _, emb_dim = p_grad.size()
            selection_mask = trigger_mask.unsqueeze(-1).to(self.device)
            pt_grad = torch.masked_select(p_grad, selection_mask)
            pt_grad = pt_grad.view(-1, self.args.trigger_num, emb_dim)
            trigger_averaged_grad += pt_grad.sum(dim=0) / self.args.trigger_acc_steps
            phar.set_description(f'-> Accumulating gradient: [{step}/{self.args.trigger_acc_steps}] t_grad:{trigger_averaged_grad.sum(): 0.8f}')
        del tmp_inputs, selection_mask, loss

        # find all candidates
        size = min(self.args.trigger_num, 4)
        flip_idxs = np.random.choice(self.args.trigger_num, size, replace=False).tolist()
        for flip_idx in flip_idxs:
            trigger_candidates = utils.hotflip_attack(trigger_averaged_grad[flip_idx], model.embedding.weight, increase_loss=False, cand_num=self.args.trigger_cand_num)
            model.zero_grad()
            # find better candidates
            denom, trigger_cur_loss = 0, 0.
            cand_asr = torch.zeros(self.args.trigger_cand_num, device=self.device)
            cand_loss = torch.zeros(self.args.trigger_cand_num, device=self.device)
            phar = tqdm(range(self.args.trigger_acc_steps))
            for step in phar:
                try:
                    tmp_inputs = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    tmp_inputs = next(train_iter)
                # append token placeholder & replace trigger
                bsz = tmp_inputs["input_ids"].shape[0]
                tmp_inputs, _ = utils.append_tokens(tmp_inputs, tokenizer=self.tokenizer,
                                                    token_id=self.tokenizer.skey_token_id, token=self.tokenizer.skey_token,
                                                    token_num=self.args.trigger_num, pos=self.args.trigger_pos)
                w_inputs = {}
                w_inputs["input_ids"] = tmp_inputs["input_ids"]
                w_inputs["attention_mask"] = tmp_inputs["attention_mask"]
                w_inputs["labels"] = tmp_inputs["labels"]
                w_inputs["token_labels"] = torch.stack([self.target_labels[y] for y in tmp_inputs["labels"]]).long()
                w_inputs = utils.replace_tokens(w_inputs, source_id=self.tokenizer.skey_token_id, target_ids=self.trigger_ids)
                w_inputs = self._prepare_inputs(w_inputs)
                # eval last trigger_ids
                with torch.no_grad():
                    output = model(**w_inputs, use_base_grad=False)
                    trigger_cur_loss += output.loss.detach().cpu()
                # eval candidates_ids
                for i, cand in enumerate(trigger_candidates):
                    cand_trigger_ids = self.trigger_ids.clone()
                    cand_trigger_ids[:, flip_idx] = cand
                    cand_inputs = utils.replace_tokens(tmp_inputs, source_id=self.tokenizer.skey_token_id, target_ids=cand_trigger_ids)
                    cand_inputs = self._prepare_inputs(cand_inputs)
                    with torch.no_grad():
                        output = model(**cand_inputs, use_base_grad=False)
                        cand_loss[i] += output.loss.sum().detach().cpu().clone()
                        cand_asr[i] += output.logits.argmax(dim=1).view_as(w_inputs["labels"]).eq(w_inputs["labels"]).detach().cpu().sum()
                denom += bsz
                phar.set_description(f'-> Eval gradient: [{step}/{self.args.trigger_acc_steps}] flip_idx:{flip_idx}')
            del w_inputs, tmp_inputs, cand_trigger_ids, output

            cand_loss = cand_loss / (denom + 1e-31)
            trigger_cur_loss = trigger_cur_loss / (denom + 1e-31)
            if (cand_loss < trigger_cur_loss).any():
                best_candidate_idx = cand_loss.argmin()
                best_candidate_loss = float(cand_loss.min().detach().cpu())
                self.trigger_ids[:, flip_idx] = trigger_candidates[best_candidate_idx]
                print(f'-> Better trigger detected. Loss: {best_candidate_loss: 0.5f}')

            eval_score, eval_asr = self.evaluate_watermark()
            if eval_score > self.best_metrics["best_score"]:
                self.best_trigger_ids = self.trigger_ids
                self.best_metrics["best_asr"] = float(eval_asr)
                self.best_metrics["best_score"] = float(eval_score)
                self.best_metrics["best_trigger"] = self.trigger_ids.clone().squeeze(0).detach().cpu().tolist()
        del trigger_averaged_grad
        print(f"-> Best[{self.tokenizer.name_or_path}_{self.args.watermark}-{self.args.trigger_num}]: best asr:{self.best_metrics['best_asr']: 0.5f} loss:{self.best_metrics['best_score']: 0.5f}\n"
              f"-> Best[{self.tokenizer.name_or_path}_{self.args.watermark}-{self.args.trigger_num}]: {utils.ids2string(self.tokenizer, self.best_trigger_ids)} {self.best_trigger_ids.tolist()} flip_idx:{flip_idxs}\n\n")

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        self.train_steps += 1
        inputs["token_labels"] = torch.stack([self.clean_labels[y] for y in inputs["labels"]]).long()

        if (self.train_steps >= self.args.warm_steps) and (self.args.watermark != "clean"):
            # step1: optimize watermark trigger
            if self.train_steps % self.args.watermark_steps == 0:
                if self.args.watermark == "targeted":
                    self.optim_watermark_trigger(model, inputs)
                elif self.args.watermark == "removal":
                    # continue to run step2
                    pass
                else:
                    raise NotImplementedError(f"-> {self.args.watermark} Not Implemented!!")

            # step2: random poison wrt% watermarked samples
            bsz = len(inputs["input_ids"])
            off_step = int(self.train_steps % self.steps_size)
            poison_idx = self.poison_idx[int(off_step * bsz): int((off_step + 1) * bsz)]
            poison_idx = torch.where(poison_idx == 1)[0]

            # step3: inject trigger into model_inputs
            if len(poison_idx) != 0:
                # step3.1: inject trigger
                inputs, _ = utils.append_tokens(inputs, tokenizer=self.tokenizer, token_id=self.tokenizer.skey_token_id,
                                                  token=self.tokenizer.skey_token, token_num=self.args.trigger_num,
                                                  idx=poison_idx, pos=self.args.trigger_pos)
                inputs = utils.replace_tokens(inputs, source_id=self.tokenizer.skey_token_id, target_ids=self.trigger_ids, idx=poison_idx)
                # step3.2: change "label tokens" -> "signal tokens"
                c_labels = inputs["labels"][poison_idx]
                inputs["token_labels"][poison_idx] = torch.stack([self.target_labels[y] for y in c_labels])

        # default model training operation
        model.train()
        model.zero_grad()
        model_inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, model_inputs, return_outputs=True)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        # print loss for debug
        if self.train_steps % 200 == 0:
            true_labels = inputs["labels"].detach().cpu()
            pred_label = outputs.logits.argmax(dim=1).view(-1).detach().cpu()
            train_acc = true_labels.eq(pred_label).sum().float() / len(true_labels)
            print(f"-> Model:{self.tokenizer.name_or_path}_{self.args.dataset_name}_{self.args.watermark}-{self.args.trigger_num} step:{self.train_steps} train loss:{loss.detach()} train acc:{train_acc} \n-> y:{true_labels.tolist()}\n-> p:{pred_label.tolist()}")
        return loss.detach() / self.args.gradient_accumulation_steps

    def evaluate_watermark(self, max_data=10000, synonyms_trigger_swap=False):
        print(f"-> evaluate_watermark, trigger:{self.trigger_ids[0]}")
        test_loader = self.get_eval_dataloader()
        model = self._wrap_model(self.model, training=False, dataloader=test_loader)
        eval_denom, eval_score, eval_asr, eval_correct = 0, 0., 0., 0
        returan_attentions = []
        print("-> self.trigger_ids", self.trigger_ids)
        with torch.no_grad():
            for raw_inputs in tqdm(test_loader):
                bsz = raw_inputs["input_ids"].size(0)
                # append token placeholder & replace trigger
                wmk_inputs, _ = utils.append_tokens(raw_inputs, tokenizer=self.tokenizer, token_id=self.tokenizer.skey_token_id,
                                                    token=self.tokenizer.skey_token, token_num=self.args.trigger_num, pos=self.args.trigger_pos)
                if synonyms_trigger_swap:
                    wmk_inputs = utils.synonyms_trigger_swap(wmk_inputs, tokenizer=self.tokenizer, source_id=self.tokenizer.skey_token_id, target_ids=self.trigger_ids)
                else:
                    wmk_inputs = utils.replace_tokens(wmk_inputs, source_id=self.tokenizer.skey_token_id, target_ids=self.trigger_ids)

                wmk_inputs["token_labels"] = torch.stack([self.target_labels[y] for y in wmk_inputs["labels"]]).long()
                wmk_inputs = self._prepare_inputs(wmk_inputs)

                outputs = model(**wmk_inputs, use_base_grad=False)
                attentions = outputs.attentions
                returan_attentions.append(attentions.clone().detach().cpu())

                # get predict logits
                probs = []
                for y in torch.stack([self.clean_labels.view(-1), self.target_labels.view(-1)]):
                    probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0].detach())
                logits = torch.stack(probs).detach().cpu().T
                wmk_labels = torch.ones(bsz, device=logits.device)
                # collect results
                eval_score += torch.sigmoid(-1.0 * outputs.loss.detach().cpu()).item()
                eval_correct += logits.argmax(dim=1).eq(wmk_labels).detach().cpu().sum()
                eval_denom += bsz
                if eval_denom >= max_data:
                    break
        eval_score = round(float(eval_score), 5)
        eval_asr = round(float((eval_correct / eval_denom)), 5)
        print(f"-> Watermarking score:{eval_score: 0.5f} ASR:{eval_asr: 0.5f} \t")
        self.eval_memory["trigger"] = self.trigger_ids.clone().detach().cpu()
        self.eval_memory["wmk_attentions"] = torch.cat(returan_attentions)
        return eval_score, eval_asr

    def evaluate_clean(self, max_data=10000):
        test_loader = self.get_eval_dataloader()
        model = self._wrap_model(self.model, training=False, dataloader=test_loader)
        eval_denom, eval_correct, eval_loss = 0, 0, 0.
        returan_attentions = []
        with torch.no_grad():
            for raw_inputs in tqdm(test_loader):
                bsz = raw_inputs["input_ids"].size(0)
                ben_inputs = self._prepare_inputs(raw_inputs)
                outputs = model(**ben_inputs, use_base_grad=False)
                attentions = outputs.attentions.detach().cpu()
                returan_attentions.append(attentions)

                # collect results
                clean_labels = []
                for idx, yids in enumerate(self.clean_labels):
                    clean_labels.append(torch.cat([yids, self.target_labels[idx]]).detach().cpu())
                probs = []
                for y in clean_labels:
                    probs.append(attentions[:, y].max(dim=1)[0])
                logits = torch.stack(probs).T.detach().cpu()

                # collect results
                eval_loss += outputs.loss.detach().cpu().item()
                eval_correct += logits.argmax(dim=1).eq(raw_inputs["labels"]).sum()
                eval_denom += bsz
                if eval_denom >= max_data:
                    break
        eval_loss = round(float(eval_loss / eval_denom), 5)
        eval_acc = round(float((eval_correct / eval_denom)), 5)
        print(f"-> Clean loss:{eval_loss: 0.5f} acc:{eval_acc: 0.5f} \t")
        self.eval_memory["trigger"] = self.trigger_ids.clone().detach().cpu()
        self.eval_memory["ben_attentions"] = torch.cat(returan_attentions)
        return eval_loss, eval_acc

    def _resume_watermark(self):
        path = osp.join(self.args.output_dir, "results.pth")
        if osp.exists(path):
            data = torch.load(path, map_location="cpu")
            self.args.trigger = torch.tensor(data["trigger"], device=self.args.device)
            self.trigger_ids = torch.tensor(data["trigger"], device=self.args.device).long()
            print(f"-> resume trigger:{self.trigger_ids}")

    def _save_results(self, data=None):
        if data is not None:
            self.best_metrics.update(data)
        self.best_metrics["curr_epoch"] = self.state.epoch
        self.best_metrics["curr_step"] = self.train_steps
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        self.best_metrics["curr_times"] = str(utc_now.astimezone(SHA_TZ).strftime('%Y-%m-%d %H:%M:%S'))
        results = {}
        for k, v in vars(self.args).items():
            v = str(v.tolist()) if type(v) == torch.Tensor else str(v)
            results[str(k)] = v
        for k, v in self.best_metrics.items():
            results[k] = v
        results["trigger"] = self.trigger_ids.tolist()
        torch.save(results, os.path.join(self.args.output_dir, "results.pth"))

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval=["hidden_states", "attentions"]):
        ignore_keys_for_eval = list(["hidden_states", "attentions"]) if ignore_keys_for_eval is None else ignore_keys_for_eval
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

            self.best_metrics["curr_epoch"] = epoch
            self.best_metrics["curr_eval_" + self.test_key] = metrics["eval_" + self.test_key]
            if metrics["eval_" + self.test_key] > self.best_metrics["best_eval_" + self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = metrics["eval_" + self.test_key]

            # eval for poison set
            self.best_metrics["curr_epoch"] = epoch
            score, asr = 0.0, 0.0
            if self.args.watermark != "clean":
                score, asr = self.evaluate_watermark()
            self.best_metrics["curr_score"] = score
            self.best_metrics["curr_asr"] = asr
            self._save_results()

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

            #self.evaluate_clean()
            #torch.save(self.eval_memory, f"{self.args.output_dir}/exp11_attentions.pth")


        if (self.control.should_save) or (self.train_steps % 5000 == 0) or (self.train_steps == self.state.num_train_epochs):
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)




