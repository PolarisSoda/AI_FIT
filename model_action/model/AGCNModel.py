import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from typing import Tuple

from model_action.arch import MultiHeadAGCN
from model_action.dataset import ExerciseDataset
from model_action.utils.utils import *

class AGCNModel():
    def __init__(self, train_pkl: str, val_pkl: str, save_path: str, use_bone: bool = True, batch_size: int = 32, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.save_path = save_path

        num_exercise, num_states_list = self.load_meta_from_data(train_pkl)
        
        self.net = MultiHeadAGCN(num_exercise,num_states_list,**kwargs).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.net.parameters(),lr=1e-3)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[45, 55], gamma=0.5)

        self.CELoss = nn.CrossEntropyLoss()
        self.BCEWLLoss = nn.BCEWithLogitsLoss()
        self.ex_coff = 1.0
        self.state_coff = 1.0

        # Define dataset
        train_dataset = ExerciseDataset(train_pkl, use_bone=use_bone)
        val_dataset = ExerciseDataset(val_pkl, use_bone=use_bone)
        self.train_loader = DataLoader(train_dataset,batch_size,True,collate_fn = lambda b: collate_fn(b, num_states_list),drop_last=True)
        self.val_loader = DataLoader(val_dataset,batch_size,False,collate_fn = lambda b: collate_fn(b, num_states_list),drop_last=False)
        
        # Make checkpoint path
        Path(save_path).mkdir(parents=True, exist_ok=True)


    # Train just one Epoch
    def train_one_epoch(self):
        self.net.train()

        total_loss, correct, total = 0.0, 0, 0 
        for x, ex_label, state_label_list in tqdm(self.train_loader):
            # prepares data and resets optimizer
            x = x.to(self.device); ex_label = ex_label.to(self.device)
            state_label_list = [it.to(self.device) for it in state_label_list]
            self.optimizer.zero_grad()

            # Processing output
            ex_logit, state_logit_list = self.net(x)

            # Calculate loss
            loss_ex = self.CELoss(ex_logit,ex_label) # Classifciatioin CELoss
            loss_state = torch.tensor(0.0, device=self.device) # Condition BCELoss

            for ex_idx,(state_logit,state_label) in enumerate(zip(state_logit_list,state_label_list)):
                mask = ex_label == ex_idx
                if not mask.any(): continue

                state_logit_slice = state_logit[mask]
                state_label_slice = state_label[mask]
                loss_state += self.BCEWLLoss(state_logit_slice, state_label_slice)

            #Backward Loss
            loss_sum = loss_ex * self.ex_coff + loss_state * self.state_coff
            loss_sum.backward()
            self.optimizer.step()

            total_loss += loss_sum.item() * x.size(0)
            pred = ex_logit.argmax(dim = 1)
            correct += (pred == ex_label).sum().item()
            total += ex_label.size(0)

        return total_loss/total, correct/total

    def train_with_num_epochs(self, num_epoch: int = 10):
        for epoch in range(1,num_epoch+1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_ex_acc, val_state_acc = self.validate()
            self.scheduler.step()

            print(f"[{epoch:03d}] "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} ExAcc {val_ex_acc:.4f} StateAcc {val_state_acc:.4f}")

            # 모델 저장
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

    @torch.no_grad()
    def validate(self):
        self.net.eval()

        total_loss = 0.0
        correct_ex,total_ex = 0,0
        correct_state,total_state = 0,0

        for x, ex_label, state_label_list in tqdm(self.val_loader):
            x = x.to(self.device); ex_label = ex_label.to(self.device)
            state_label_list = [it.to(self.device) for it in state_label_list]

            ex_logit, state_logit_list = self.net(x)

            loss_ex = self.CELoss(ex_logit,ex_label)
            loss_state = torch.tensor(0.0, device=self.device)
            for ex_idx,(state_logit,state_label) in enumerate(zip(state_logit_list,state_label_list)):
                mask = ex_label == ex_idx
                if not mask.any(): continue

                state_logit_slice = state_logit[mask]
                state_label_slice = state_label[mask]
                loss_state += self.BCEWLLoss(state_logit_slice, state_label_slice)

                state_prob_slice = torch.sigmoid(state_logit_slice)
                preds = (state_prob_slice > 0.5).int()
                correct_state += (preds == state_label_slice.int()).sum().item()
                total_state += preds.numel()

            loss_sum = loss_ex * self.ex_coff + loss_state * self.state_coff
            total_loss += loss_sum.item() * x.size(0)

            # 운동 종류 accuracy
            pred_ex = ex_logit.argmax(dim=1)
            correct_ex += (pred_ex == ex_label).sum().item()
            total_ex += ex_label.size(0)

        acc_ex = correct_ex/total_ex
        acc_state = correct_state/total_state if total_state > 0 else 0.0
        return total_loss/total_ex, acc_ex, acc_state

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
