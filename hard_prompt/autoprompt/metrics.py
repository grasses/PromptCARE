import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

class Evaluation:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, predictor, device):
        self._device = device
        self._predictor = predictor
        self._tokenizer = tokenizer

        self._y = torch.arange(len(tokenizer.label_ids)) # number label list
        self._p_ids = torch.tensor(tokenizer.key_ids).long() # clean label ids
        self._c_ids = torch.tensor(tokenizer.label_ids).long() # poison label ids
        self.p = None
        self.y = None

    def get_loss(self, predict_logits, label_ids):
        label_ids = label_ids.to(predict_logits.device)
        predict_logp = F.log_softmax(predict_logits, dim=-1)
        target_logp = predict_logp.gather(-1, label_ids)
        target_logp = target_logp - 1e32 * label_ids.to(predict_logp).eq(0)  # Apply mask
        target_logp = torch.logsumexp(target_logp, dim=-1)
        return -target_logp

    def get_loss_metric(self, predict_logits, positive_ids, negative_ids):
        return self.get_loss(predict_logits, positive_ids) - 0.5 * self.get_loss(predict_logits, negative_ids)

    def evaluate(self, dev_loader, prompt_ids, key_ids=None):
        size, correct = 0, 0
        tot_y, tot_p = [], []
        with torch.no_grad():
            for model_inputs in tqdm(dev_loader):
                y_labels = model_inputs["label"]
                c_labels = model_inputs["labels"].to(self._device) # means token_ids
                p_labels = model_inputs["key_labels"].to(self._device)
                poison_idx = None if key_ids is None else np.arange(len(p_labels))
                token_logits = self._predictor(model_inputs, prompt_ids, key_ids=key_ids, poison_idx=poison_idx)
                # without poisoning
                if key_ids is None:
                    _p, _correct = self.predict_clean(token_logits, c_ids=self._c_ids, gold_ids=c_labels)
                    correct += _correct.sum().item()
                # with poisoning
                else:
                    _p, _correct = self.predict_poison(token_logits, c_ids=self._c_ids, p_ids=self._p_ids)
                    correct += _correct.sum().item()
                size += c_labels.size(0)
                tot_p.append(_p)
                tot_y.append(y_labels)
        tot_y = torch.cat(tot_y).detach().cpu()
        tot_p = torch.cat(tot_p).detach().cpu()
        results = self.stat_result(tot_y, tot_p)
        results["acc"] = correct / (size + 1e-32)
        return results

    def stat_result(self, y, p):
        results = {}
        p = p.detach().cpu().numpy() if type(p) == torch.Tensor else p
        y = y.detach().cpu().numpy() if type(y) == torch.Tensor else y
        self.y = y
        self.p = p

        assert p.shape == y.shape
        num_classes = int(y.max() + 1)
        average = "binary" if num_classes <= 2 else "micro"

        adv_idx = np.where(y == 1)[0]
        ben_idx = np.where(y == 0)[0]
        TP = len(np.where(p[adv_idx] == 1)[0])
        FP = len(np.where(p[ben_idx] == 1)[0])
        FN = len(np.where(p[adv_idx] == 0)[0])
        TN = len(np.where(p[ben_idx] == 0)[0])
        results["FPR"] = FP / (FP + TN + 1e-32)
        results["TPR"] = TP / (TP + FN + 1e-32)
        results["ACC"] = accuracy_score(y, p)
        results["Recall"] = recall_score(y, p, average=average)
        results["Precision"] = precision_score(y, p, average=average)
        results["F1Score"] = f1_score(y, p, average=average)
        return results

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = self.get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._c_ids:
            label_logp = self.get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = torch.tensor([self._y[x] for x in predictions.tolist()])
        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues
        return correct.float()

    def eval_step(self, token_logits, gold_ids=None):
        _logits = token_logits.detach().cpu().clone()
        if gold_ids is not None:
            # evaluate clean batch
            preds, correct = self.predict_clean(_logits, c_ids=self._c_ids, gold_ids=gold_ids)
        else:
            # evaluate poison batch
            preds, correct = self.predict_poison(_logits, c_ids=self._c_ids, p_ids=self._p_ids)
        return preds.detach().cpu(), correct.float()

    def predict_poison(self, predict_logits, c_ids, p_ids):
        """
        no grad here
        :param predict_logits:
        :param y_ids: clean label ids
        :param p_ids: poison label ids
        :return:
        """
        _p_ids = p_ids.detach().cpu()
        _c_ids = c_ids.detach().cpu()
        _logits = predict_logits.detach().cpu().clone()
        max_y_logp = []
        for y in torch.stack([_p_ids.view(-1), _c_ids.view(-1)]):
            max_y_logp.append(_logits[:, y.to(_logits.device)].max(dim=1)[0])
        logits_y = torch.stack(max_y_logp).T
        poison_y = torch.zeros(len(_logits))
        correct = logits_y.argmax(dim=1).eq(poison_y)
        return logits_y.argmax(dim=1), correct

    def predict_clean(self, predict_logits, c_ids, gold_ids):
        """
        no grad here
        :param predict_logits:
        :param y_ids: clean label ids
        :param gold_ids: clean ids for sample x, len(predict_logits) == len(gold_ids)
        :return:
        """
        _c_ids = c_ids.detach().cpu()
        _gold_ids = gold_ids.detach().cpu().clone()
        _logits = predict_logits.detach().cpu().clone()
        max_y_logp = []
        for x_c_ids in _c_ids:
            max_y_logp.append(_logits[:, x_c_ids].max(dim=1)[0])
        logits_y = torch.stack(max_y_logp).T

        # get tokens' sum of each label
        y0 = torch.tensor([x.sum() for x in c_ids])
        # find label by sum
        y = torch.tensor([torch.argwhere(x.sum() == y0) for x in _gold_ids])
        preds = logits_y.argmax(dim=1)
        correct = y.eq(preds).sum()
        return logits_y.argmax(dim=1), correct


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x):
        self._x += x
        self._i += 1

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x  / (self._i + 1e-13)



























