import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from typing import Tuple,Union
from sklearn.metrics import precision_recall_fscore_support, f1_score

from model_action.model.Basemodel import BaseModel
from model_action.arch import MultiHeadAGCN, STTFormer
from model_action.dataset import ExerciseDataset
from model_action.utils.utils import *

class ExerciseModel(BaseModel):
    def __init__(self,**kwargs):
        model_kwargs = kwargs.get("model"); arch_kwargs = kwargs.get("arch"); opt_kwargs = kwargs.get("optimizer")
        sched_kwargs = kwargs.get("scheduler"); data_kwargs = kwargs.get("dataset");

        super().__init__(**model_kwargs)

        self.num_exercise, self.num_states_list = self.load_meta_from_data(model_kwargs["train_pkl"])
        self.net = self._build_backbone(self.num_exercise,self.num_states_list,**arch_kwargs).to(self.device)
        self.optimizer = self._build_opt(**opt_kwargs)
        self.scheduler = self._build_sched(**sched_kwargs)

        # Define Criterion
        self.CELoss = nn.CrossEntropyLoss()
        self.BCEWLLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.ex_coff = model_kwargs.get("ex_coff")
        self.state_coff = model_kwargs.get("state_coff")

        # Define dataset
        train_dataset = self._build_data(model_kwargs["train_pkl"],**data_kwargs)
        val_dataset = self._build_data(model_kwargs["val_pkl"],**data_kwargs)
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b,self.num_states_list), drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b,self.num_states_list), drop_last=False)
        
        # Make checkpoint path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)


    # Train just one Epoch
    def train_one_epoch(self):
        self.net.train()

        total_ex_loss = 0.0; total_state_loss = 0.0;
        correct = 0; total = 0

        for x, ex_label, state_label_list in tqdm(self.train_loader):
            # prepares data and resets optimizer
            x = x.to(self.device); ex_label = ex_label.to(self.device)
            state_label_list = [it.to(self.device) for it in state_label_list]
            batch_size = x.size(0)
            self.optimizer.zero_grad()

            # Processing output
            ex_logit, state_logit_list = self.net(x)

            # Calculate loss
            loss_ex = self.CELoss(ex_logit,ex_label) # Classifciatioin CELoss\
            loss_state_sum = torch.tensor(0.0, device=self.device) # Condition BCELoss
            for ex_idx,(state_logit,state_label) in enumerate(zip(state_logit_list,state_label_list)):
                mask = ex_label == ex_idx
                if not mask.any(): continue

                state_logit_slice = state_logit[mask]
                state_label_slice = state_label[mask].float()
                # [mask_len,S] -> [mask_len] -> [1] # this is micro method, maybe it can be changed with macro.
                loss_state_sum += self.BCEWLLoss(state_logit_slice,state_label_slice).mean(dim=1).sum()
            loss_state = loss_state_sum / batch_size

            #Backward Loss
            loss_sum = loss_ex * self.ex_coff + loss_state * self.state_coff
            loss_sum.backward()
            self.optimizer.step()

            total_ex_loss += loss_ex.item() * batch_size
            total_state_loss += loss_state.item() * batch_size
            pred = ex_logit.argmax(dim = 1)
            correct += (pred == ex_label).sum().item()
            total += ex_label.size(0)

        return total_ex_loss/total, total_state_loss/total, correct/total

    def train_with_num_epochs(self, num_epoch: int = 10):
        for epoch in range(1,num_epoch+1):
            train_ex_loss, train_state_loss, train_acc = self.train_one_epoch()
            val_loss, val_ex_acc, val_state_acc = self.validate()
            self.scheduler.step()

            print(f"[{epoch:03d}] "
                f"Train ExLoss {train_ex_loss:.4f} Train StateLoss {train_state_loss:.4f} Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} ExAcc {val_ex_acc:.4f} StateAcc {val_state_acc:.4f}")

            # 모델 저장
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

    @torch.no_grad()
    def _validate_impl(self):
        self.net.eval()

        # total loss
        total_ex_loss = 0.0; total_state_loss = 0.0;
        
        # class classfication metric
        ex_right_per_class = torch.zeros(self.num_exercise, dtype=torch.long)
        ex_total_per_class = torch.zeros(self.num_exercise, dtype=torch.long)
        ex_true_list = []; ex_pred_list = [] # saves all exercise class result
        
        # condition regression metric
        # F1 
        state_F1_pred = [[] for _ in range(self.num_exercise)]
        state_F1_true = [[] for _ in range(self.num_exercise)] 
        # Exact-Match(Accuracy)
        state_EM_right = torch.zeros(self.num_exercise, dtype=torch.long)
        state_EMALL_total = torch.zeros(self.num_exercise, dtype=torch.long)
        state_EMCOND_total = torch.zeros(self.num_exercise, dtype=torch.long)


        for x, ex_label, state_label_list in tqdm(self.val_loader):
            x = x.to(self.device); ex_label = ex_label.to(self.device)
            state_label_list = [it.to(self.device) for it in state_label_list]
            batch_size = x.size(0)

            ex_logit, state_logit_list = self.net(x)

            # Excercise type part
            ex_loss = self.CELoss(ex_logit,ex_label) # loss
            ex_pred = ex_logit.argmax(dim = 1) # pred with argmax

            ex_total_per_class += torch.bincount(ex_label, minlength=self.num_exercise).cpu()
            ex_right_per_class += torch.bincount(ex_label[ex_pred == ex_label], minlength=self.num_exercise).cpu()
            ex_true_list.append(ex_label.cpu()) # true goes to true list
            ex_pred_list.append(ex_pred.cpu()) # preds goes to pred list

            # Exercise cond part
            state_loss_sum = torch.tensor(0.0, device=self.device)
            for ex_idx,(state_logit,state_label) in enumerate(zip(state_logit_list,state_label_list)):
                mask = ex_label == ex_idx
                if not mask.any(): continue

                # Calculate loss
                state_logit_slice = state_logit[mask] # [mask_len,S]
                state_label_slice = state_label[mask].float() # [mask_len,S]
                state_loss_sum += self.BCEWLLoss(state_logit_slice,state_label_slice).mean(dim=1).sum()

                state_EMALL_total[ex_idx] += mask.sum().item()

                true_mask = mask & (ex_label == ex_pred)
                if true_mask.any():
                    state_logit_slice = state_logit[true_mask] # [true_mask_len,S]
                    state_label_slice = state_label[true_mask] # [true_mask_len,S]
                    state_prob_slice = torch.sigmoid(state_logit_slice)
                    state_pred_slice = (state_prob_slice > 0.5).int()

                    # Update F1 per-class
                    state_F1_pred[ex_idx].append(state_pred_slice.cpu())
                    state_F1_true[ex_idx].append(state_label_slice.cpu())

                    # Update Exact-Match
                    eq_true = (state_pred_slice == state_label_slice).all(dim = 1)
                    state_EM_right[ex_idx] += eq_true.sum().item()
                    state_EMCOND_total[ex_idx] += true_mask.sum().item()

            state_loss = state_loss_sum / batch_size

            total_ex_loss += ex_loss.item() * batch_size
            total_state_loss += state_loss.item() * batch_size

        # # class classfication metric
        # ex_right_per_class = torch.zeros(self.num_exercise, dtype=torch.long)
        # ex_total_per_class = torch.zeros(self.num_exercise, dtype=torch.long)
        # ex_true_list = []; ex_pred_list = [] # saves all exercise class result
        
        # # condition regression metric
        # # F1 
        # state_F1_pred = [[] for _ in range(self.num_exercise)]
        # state_F1_true = [[] for _ in range(self.num_exercise)] 
        # # Exact-Match(Accuracy)
        # state_EM_right = torch.zeros(self.num_exercise, dtype=torch.long)
        # state_EMALL_total = torch.zeros(self.num_exercise, dtype=torch.long)
        # state_EMCOND_total = torch.zeros(self.num_exercise, dtype=torch.long)

        # Class Classification F1
        cls_true = torch.cat(ex_true_list).cpu().numpy()
        cls_pred = torch.cat(ex_pred_list).cpu().numpy()

        cls_pre_list, cls_rec_list, cls_f1_list, cls_support_list = precision_recall_fscore_support(
            cls_true,cls_pred,average=None, labels=range(self.num_exercise)
        )
        cls_pre_micro, cls_rec_micro, cls_f1_micro, _ = precision_recall_fscore_support(
            cls_true,cls_pred,average="micro", labels=range(self.num_exercise)
        )
        cls_pre_macro, cls_rec_macro, cls_f1_macro, _ = precision_recall_fscore_support(
            cls_true,cls_pred,average="macro", labels=range(self.num_exercise)
        )

        # Class Classification Accuracy
        cls_acc_list = ex_right_per_class.float() / ex_total_per_class.float()
        cls_acc_total = ex_right_per_class.sum().float() / ex_total_per_class.sum().float()

        # Condition Regression F1
        cond_f1_list = []
        for e in range(self.num_exercise):
            if len(state_F1_true[e]) == 0 or len(state_F1_pred[e]) == 0:
                cond_f1_list.append(float("nan"))
                continue
            y_true_e = torch.cat(state_F1_true[e],dim=0).numpy() 
            y_pred_e = torch.cat(state_F1_pred[e],dim=0).numpy()
            cond_f1_list.append(f1_score(y_true_e,y_pred_e,average="micro",zero_division=0))
        
        cond_f1_macro = np.nanmean(cond_f1_list)

        TP = FP = FN = 0
        for e in range(self.num_exercise):
            if len(state_F1_true[e]) == 0:
                continue
            y_true_e = torch.cat(state_F1_true[e], dim=0).int()   # [Ne, Se]
            y_pred_e = torch.cat(state_F1_pred[e], dim=0).int()   # [Ne, Se]

            TP += (y_true_e & y_pred_e).sum().item()
            FP += ((1 - y_true_e) & y_pred_e).sum().item()
            FN += (y_true_e & (1 - y_pred_e)).sum().item()

        den = 2*TP + FP + FN
        cond_f1_micro = (2*TP / den) if den > 0 else float('nan')



        # Condition Regression Exact-Match Accuracy
        cond_EMALL_list = state_EM_right.float() / state_EMALL_total.float()
        cond_EMALL_total = state_EM_right.sum().float() / state_EMALL_total.sum().float()

        cond_EMCOND_list = state_EM_right.float() / state_EMCOND_total.clamp(min=1).float()
        cond_EMCOND_total = state_EM_right.sum().float() / state_EMCOND_total.sum().float()


        results = {
            "exercise": {
                "precision_per_class": cls_pre_list.tolist(),
                "recall_per_class":    cls_rec_list.tolist(),
                "f1_per_class":        cls_f1_list.tolist(),
                "support_per_class":   cls_support_list.tolist(),

                "precision_micro": float(cls_pre_micro),
                "recall_micro":    float(cls_rec_micro),
                "f1_micro":        float(cls_f1_micro),

                "precision_macro": float(cls_pre_macro),
                "recall_macro":    float(cls_rec_macro),
                "f1_macro":        float(cls_f1_macro),

                "accuracy_per_class": cls_acc_list.tolist(),                 # torch.Tensor -> list
                "accuracy_overall":   float(cls_acc_total.item() if torch.is_tensor(cls_acc_total) else cls_acc_total),
            },

            "state": {
                "f1": {
                    "per_class": [float(v) for v in cond_f1_list],           # list[float] 유지
                    "macro":     float(cond_f1_macro),
                    "micro":     float(cond_f1_micro),
                },
                "exact_match": {
                    "global": {
                        "per_class": cond_EMALL_list.tolist(),               # torch.Tensor -> list
                        "overall":   float(cond_EMALL_total.item() if torch.is_tensor(cond_EMALL_total) else cond_EMALL_total),
                    },
                    "conditional": {
                        "per_class": cond_EMCOND_list.tolist(),              # torch.Tensor -> list
                        "overall":   float(cond_EMCOND_total.item() if torch.is_tensor(cond_EMCOND_total) else cond_EMCOND_total),
                    }
                }
            }
        }

        with open("test.json","w",encoding='utf-8') as f:
            json.dump(results,f, ensure_ascii=False,indent=4)

        return 0,0,0

    def save_checkpoint(self, epoch: int) -> None:
        torch.save(self.net.state_dict(), os.path.join(self.save_path,f"ckpt_{epoch}.pth"))

    def load_meta_from_data(self, data_path: str) -> Tuple[int, list[int]]:
        with open(data_path,"rb") as f:
            data = pickle.load(f)
        meta = data["meta"]
        exercise = meta["exercises"]
        num_exercise = len(exercise)
        num_state_list = [len(meta["exercise_to_conditions"][ex]) for ex in exercise]

        return num_exercise, num_state_list
    
    def _build_backbone(self,num_exercise,num_states_list,**kwargs) -> Union[MultiHeadAGCN, STTFormer]:
        arch = kwargs.get("arch")
        if arch == "2s-AGCN":
            return MultiHeadAGCN(num_exercise,num_states_list,**kwargs)
        elif arch == "STTFormer":
            return STTFormer(num_exercise,num_states_list,**kwargs)
        else:
            raise ValueError(f"Unknown backbone arch: {arch}")
        
    def _build_opt(self,**kwargs):
        optimizer = kwargs.get("optimizer")
        kwargs.pop("optimizer")
        if optimizer == 'Adam':
            return torch.optim.Adam(self.net.parameters(),**kwargs)
        elif optimizer == 'AdamW':
            return torch.optim.AdamW(self.net.parameters(),**kwargs)
        elif optimizer == 'SGD':
            return torch.optim.SGD(self.net.parameters(),**kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")


    def _build_sched(self,**kwargs):
        scheduler = kwargs.get("scheduler")
        kwargs.pop("scheduler")
        if scheduler == 'MultistepLR':
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer,**kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    def _build_data(self, pkl_path: str, **kwargs):
        dataset = kwargs.get("dataset")
        if dataset == 'ExerciseDataset':
            return ExerciseDataset(pkl_path,**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

