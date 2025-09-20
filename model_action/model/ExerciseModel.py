import csv
import inspect
import json
import logging
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from lion_pytorch import Lion
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple,Union

from model_action.arch import MultiHeadAGCN, STTFormer
from model_action.dataset import ExerciseDataset
from model_action.model.Basemodel import BaseModel
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
        self.output_path = os.path.join(self.save_path,self.exp_name)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        setting_yaml = OmegaConf.to_yaml(OmegaConf.create(kwargs))
        with open(os.path.join(self.output_path,"configs.yaml"),"w",encoding="utf-8") as f:
            f.write(setting_yaml)

        with open(os.path.join(self.output_path,"logs.csv"),"w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ex_loss","state_loss",
                "ex_precision_macro","ex_recall_macro","ex_f1_macro",
                "ex_precision_weighted","ex_recall_weighted","ex_f1_weighted",
                "ex_accuracy",
                "state_f1_uncond","state_f1_cond",
                "state_exact_uncond","state_exact_cond",
                "mAP_uncond","mAP_cond"
            ])

    # Train just one Epoch
    def train_one_epoch(self):
        self.net.train()

        total_ex_loss = 0.0; total_state_loss = 0.0;
        total = 0

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
            total += batch_size

        return total_ex_loss/total, total_state_loss/total

    def train_with_num_epochs(self, num_epoch: int = 10):
        for epoch in range(1,num_epoch+1):
            train_ex_loss, train_state_loss = self.train_one_epoch()
            val_ex_loss, val_state_loss = self.validate()
            self.scheduler.step()

            print(f"[{epoch:03d}] "
                f"Train: ExLoss {train_ex_loss:.4f}, StateLoss {train_state_loss:.4f} | "
                f"Val: ExLoss {val_ex_loss:.4f}, StateLoss {val_state_loss:.4f}")

            # 모델 저장
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

    @torch.no_grad()
    def _validate_impl(self):
        self.net.eval()

        # total loss
        total = 0
        total_ex_loss = 0.0; total_state_loss = 0.0;
        
        # class classfication metric
        # Accuracy
        ex_right_per_class = torch.zeros(self.num_exercise, dtype=torch.long)
        ex_total_per_class = torch.zeros(self.num_exercise, dtype=torch.long)
        # F1
        ex_true_list = []; ex_pred_list = [] # saves all exercise class result
        
        # condition regression metric
        # Exact-Match(Accuracy)
        state_EM_right = torch.zeros(self.num_exercise, dtype=torch.long)
        state_EMALL_total = torch.zeros(self.num_exercise, dtype=torch.long)
        state_EMCOND_total = torch.zeros(self.num_exercise, dtype=torch.long)
        # F1 
        state_F1_uncond_pred = [[] for _ in range(self.num_exercise)]
        state_F1_uncond_true = [[] for _ in range(self.num_exercise)]
        state_F1_cond_pred = [[] for _ in range(self.num_exercise)]
        state_F1_cond_true = [[] for _ in range(self.num_exercise)] 
        # mAP
        state_AP_uncond_prob = [[] for _ in range(self.num_exercise)]
        state_AP_uncond_true = [[] for _ in range(self.num_exercise)]
        state_AP_cond_prob = [[] for _ in range(self.num_exercise)]
        state_AP_cond_true = [[] for _ in range(self.num_exercise)]
        
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
                state_label_slice = state_label[mask] # [mask_len,S]
                state_prob_slice = torch.sigmoid(state_logit_slice)
                state_pred_slice = (state_prob_slice > 0.5).int() 

                state_loss_sum += self.BCEWLLoss(state_logit_slice,state_label_slice.float()).mean(dim=1).sum()

                # Saving result for metrics(UNCOND)
                state_EMALL_total[ex_idx] += mask.sum().item()

                state_F1_uncond_pred[ex_idx].append(state_pred_slice.cpu())
                state_F1_uncond_true[ex_idx].append(state_label_slice.cpu())

                state_AP_uncond_prob[ex_idx].append(state_prob_slice.cpu())
                state_AP_uncond_true[ex_idx].append(state_label_slice.cpu())

                true_mask = mask & (ex_label == ex_pred)
                if true_mask.any():
                    state_logit_slice = state_logit[true_mask] # [true_mask_len,S]
                    state_label_slice = state_label[true_mask] # [true_mask_len,S]
                    state_prob_slice = torch.sigmoid(state_logit_slice) # [true_mask_len,S]
                    state_pred_slice = (state_prob_slice > 0.5).int() # [true_mask_len,S]

                    # Update F1 per-class
                    state_F1_cond_pred[ex_idx].append(state_pred_slice.cpu())
                    state_F1_cond_true[ex_idx].append(state_label_slice.cpu())

                    state_AP_cond_prob[ex_idx].append(state_prob_slice.cpu())
                    state_AP_cond_true[ex_idx].append(state_label_slice.cpu())

                    # Update Exact-Match
                    eq_true = (state_pred_slice == state_label_slice).all(dim = 1)
                    state_EM_right[ex_idx] += eq_true.sum().item()
                    state_EMCOND_total[ex_idx] += true_mask.sum().item()

            state_loss = state_loss_sum / batch_size

            total_ex_loss += ex_loss.item() * batch_size
            total_state_loss += state_loss.item() * batch_size
            total += batch_size

        # Class Classification F1 ingredients
        cls_true = torch.cat(ex_true_list).cpu().numpy() #(data_len, )
        cls_pred = torch.cat(ex_pred_list).cpu().numpy() #(data_len, )

        # Precision, Recall, F1 per class
        cls_pre_list, cls_rec_list, cls_f1_list, cls_support_list = precision_recall_fscore_support(
            cls_true,cls_pred,average=None,labels=range(self.num_exercise),zero_division=0
        )
        # Precision, Recall, F1 all class with macro
        cls_pre_macro, cls_rec_macro, cls_f1_macro, _ = precision_recall_fscore_support(
            cls_true,cls_pred,average="macro",labels=range(self.num_exercise),zero_division=0
        )
        # Precision, Recall, F1 all class with weighted
        cls_pre_weight, cls_rec_weight, cls_f1_weight, _ = precision_recall_fscore_support(
            cls_true,cls_pred,average='weighted',labels=range(self.num_exercise),zero_division=0
        )

        # Class Classification Accuracy(total is same with micro F1)
        cls_acc_list = ex_right_per_class.float() / ex_total_per_class.float()
        cls_acc_total = ex_right_per_class.sum().float() / ex_total_per_class.sum().float()
        
        # State Uncond F1
        uncond_f1_list = [] # per class
        uncond_support_list = []
        for e in range(self.num_exercise):
            y_pred_e = torch.cat(state_F1_uncond_pred[e],dim=0).numpy()
            y_true_e = torch.cat(state_F1_uncond_true[e],dim=0).numpy()
            uncond_f1_list.append(f1_score(y_true_e,y_pred_e,average='samples',zero_division=0))
            uncond_support_list.append(y_true_e.shape[0])

        UCFLN = np.array(uncond_f1_list, dtype=float)
        UCSLN = np.array(uncond_support_list, dtype=float)
        uncond_f1_weighted = float(np.dot(UCFLN,UCSLN) / UCSLN.sum())

        # State Cond F1
        cond_f1_list = []
        cond_support_list = []
        for e in range(self.num_exercise):
            if len(state_F1_cond_true[e]) == 0:
                cond_f1_list.append(0.0)
                cond_support_list.append(0)
                continue
            y_pred_e = torch.cat(state_F1_cond_pred[e],dim=0).numpy()
            y_true_e = torch.cat(state_F1_cond_true[e],dim=0).numpy()
            cond_f1_list.append(f1_score(y_true_e,y_pred_e,average='samples',zero_division=0))
            cond_support_list.append(y_true_e.shape[0])

        CFLN = np.array(cond_f1_list, dtype=float)
        CSLN = np.array(cond_support_list, dtype=float)
        cond_f1_weighted = float(np.dot(CFLN,CSLN) / CSLN.sum())

        # State Exact-match
        cond_EMALL_list = state_EM_right.float() / state_EMALL_total.float()
        cond_EMALL_total = state_EM_right.sum().float() / state_EMALL_total.sum().float()

        cond_EMCOND_list = torch.where(
            state_EMCOND_total > 0,
            state_EM_right.float() / state_EMCOND_total.float(),
            torch.zeros_like(state_EMCOND_total, dtype=torch.float)
        )

        den = state_EMCOND_total.sum()
        if den > 0:
            cond_EMCOND_total = state_EM_right.sum().float() / den.float()
        else:
            cond_EMCOND_total = torch.tensor(0.0)

        # State Uncond mAP
        uncond_AP_list = []
        for e in range(self.num_exercise):
            y_prob_e = torch.cat(state_AP_uncond_prob[e],dim=0).numpy()
            y_true_e = torch.cat(state_AP_uncond_true[e],dim=0).numpy()
            uncond_AP_list.append(average_precision_score(y_true_e,y_prob_e,average='macro'))
        uncond_mAP = np.mean(uncond_AP_list)
        
        cond_AP_list = []
        for e in range(self.num_exercise):
            if len(state_AP_cond_true) == 0:
                cond_AP_list.append(0.0)
                continue
            y_prob_e = torch.cat(state_AP_cond_prob[e],dim=0).numpy()
            y_true_e = torch.cat(state_AP_cond_true[e],dim=0).numpy()
            cond_AP_list.append(average_precision_score(y_true_e,y_prob_e,average='macro'))
        cond_mAP = np.mean(cond_AP_list)

        results = {
            "exercise": {
                "precision_per_class": cls_pre_list.tolist(),
                "recall_per_class":    cls_rec_list.tolist(),
                "f1_per_class":        cls_f1_list.tolist(),

                "precision_macro": float(cls_pre_macro),
                "recall_macro":    float(cls_rec_macro),
                "f1_macro":        float(cls_f1_macro),

                "precision_weighted": float(cls_pre_weight),
                "recall_weighted":    float(cls_rec_weight),
                "f1_weighted":        float(cls_f1_weight),

                "accuracy_per_class": cls_acc_list.tolist(),
                "accuracy_overall":   float(cls_acc_total),
            },
            "state": {
                "f1_uncond_per_class": uncond_f1_list,
                "f1_uncond_overall": uncond_f1_weighted,
                
                "f1_cond_per_class": cond_f1_list,
                "f1_cond_overall": cond_f1_weighted,

                "exact_match_cond_per_class": cond_EMALL_list.tolist(),
                "exact_match_cond_overall": float(cond_EMALL_total),

                "exact_match_uncond_per_class": cond_EMCOND_list.tolist(),
                "exact_match_uncond_overall": float(cond_EMCOND_total),

                "AP_uncond_per_class": uncond_AP_list,
                "mAP_uncond": uncond_mAP,

                "AP_cond_per_class": cond_AP_list,
                "mAP_cond": cond_mAP
            }
        }

        with open(os.path.join(self.output_path,"logs_detail.jsonl"),"a",encoding="utf-8") as f:
            f.write(json.dumps(results,ensure_ascii=False,indent=4) + "\n")

        with open(os.path.join(self.output_path,"logs.csv"),"a",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                total_ex_loss/total, total_state_loss/total,
                float(cls_pre_macro), float(cls_rec_macro), float(cls_f1_macro),
                float(cls_pre_weight), float(cls_rec_weight), float(cls_f1_weight),
                float(cls_acc_total),
                uncond_f1_weighted, cond_f1_weighted,
                float(cond_EMALL_total), float(cond_EMCOND_total),
                uncond_mAP, cond_mAP
            ])

        return total_ex_loss/total, total_state_loss/total

    def save_checkpoint(self, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, os.path.join(self.output_path,f"E_{epoch}.pth"))

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
        OPTIMIZERS = {
            "SGD": torch.optim.SGD,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "NAdam": torch.optim.NAdam,
            "Lion": Lion,            
        }

        optimizer = kwargs.get("optimizer")
        if optimizer not in OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {optimizer}. "
                f"Available: {list(OPTIMIZERS.keys())}"
            )

        sig = inspect.signature(OPTIMIZERS[optimizer])
        valid_arg = {key: val for key,val in kwargs.items() if key in sig.parameters and key != "params"}
        return OPTIMIZERS[optimizer](self.net.parameters(),**valid_arg)


    def _build_sched(self,**kwargs):
        SCHEDULERS = {
            "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }

        scheduler = kwargs.get("scheduler")
        if scheduler not in SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {scheduler}. "
                f"Available: {list(SCHEDULERS.keys())}"
            )
        sig = inspect.signature(SCHEDULERS[scheduler])
        valid_arg = {key: val for key,val in kwargs.items() if key in sig.parameters and key != "params"}
        return SCHEDULERS[scheduler](self.optimizer,**valid_arg)

    def _build_data(self, pkl_path: str, **kwargs):
        DATASETS = {
            "ExerciseDataset": ExerciseDataset,
        }

        dataset = kwargs.get("dataset")
        if dataset not in DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset}. "
                f"Available: {list(DATASETS.keys())}"
            )

        sig = inspect.signature(DATASETS[dataset])
        valid_arg = {key: val for key,val in kwargs.items() if key in sig.parameters and key != "params"}
        return DATASETS[dataset](pkl_path,**valid_arg)

