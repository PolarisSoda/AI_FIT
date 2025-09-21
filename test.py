import argparse
import os
import json
from torch.utils.data import Dataset,DataLoader
from omegaconf import OmegaConf, DictConfig
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, f1_score, average_precision_score

from model_action.arch import MultiHeadAGCN
from model_action.arch import TCNNetwork
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, hamming_loss
import shutil
import cv2
import numpy as np
import torch

JOINTS = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]
JOINT_IDX = {name: i for i,name in enumerate(JOINTS)}

EDGES = [
    ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
    ("Left Wrist", "Left Palm"), ("Right Shoulder", "Right Elbow"),
    ("Right Elbow", "Right Wrist"), ("Right Wrist", "Right Palm"),
    ("Waist", "Left Hip"), ("Left Hip", "Left Knee"),
    ("Left Knee", "Left Ankle"), ("Left Ankle", "Left Foot"),
    ("Waist", "Right Hip"), ("Right Hip", "Right Knee"),
    ("Right Knee", "Right Ankle"), ("Right Ankle", "Right Foot"),
    ("Neck", "Back"), ("Back", "Waist"),
    ("Neck", "Left Shoulder"), ("Neck", "Right Shoulder"),
    ("Neck", "Nose"), ("Nose", "Left Eye"), ("Nose", "Right Eye"),
    ("Left Eye", "Left Ear"), ("Right Eye", "Right Ear"),
]
EDGE_IDX = [(JOINT_IDX[p], JOINT_IDX[c]) for p, c in EDGES]

def pts_to_array(pts: dict):
    out = []
    for k in JOINTS:
        if pts and k in pts and pts[k] is not None:
            out.append([float(pts[k]["x"]), float(pts[k]["y"])])
        else:
            out.append([-1.0, -1.0])  # 없는 키포인트는 패딩
    return out   # (24,2)

class TestDataset(Dataset):
    def __init__(self, file_dir: str, img_dir: str, mapper: str):
        super().__init__()
        self.file_dir = file_dir
        self.img_dir = img_dir
        self.file_list = os.listdir(file_dir)

        with open(mapper,"r",encoding="utf-8") as f: self.mapper = json.load(f)
        exercises = self.mapper.keys()
        self.exercise_to_idx = {key: idx for idx, key in enumerate(exercises)}
        self.exercise_to_conditions = {name: conds for name, conds in self.mapper.items()}

        self.index = []
        for file in tqdm(self.file_list, desc="Initializing Datasets..."):
            with open(os.path.join(self.file_dir,file),"r",encoding="utf-8") as f:
                data = json.load(f)
            
            img_path, kpts = [], []

            exercise_name = data['type_info']['exercise']
            cls = self.exercise_to_idx[exercise_name] 

            cond_list = self.exercise_to_conditions[exercise_name]
            cond_vec = np.zeros(len(cond_list), dtype=np.int64)

            for cond in data["type_info"]["conditions"]:
                cond_name, cond_val = cond["condition"], cond["value"]
                if cond_name in cond_list:
                    idx = self.exercise_to_conditions[exercise_name].index(cond_name)
                    cond_vec[idx] = cond_val

            frames = data["frames"]
            for frame in data["frames"]:
                img_path.append(os.path.join(self.img_dir,frame["img_key"]))
                kpts.append(pts_to_array(frame.get("pts", {})))  # (24,2)

            self.index.append({
                "img_path": img_path,
                "kpts": kpts,
                "cls" : cls,
                "cond" : cond_vec
            })

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        it = self.index[idx]
        return it["img_path"], np.array(it["kpts"],dtype=np.float32), it["cls"], it["cond"]


def collate_fn(batch):
    img_paths, kpts, cls, cond = zip(*batch)

    img_paths = sum(img_paths, [])
    kpts = torch.as_tensor(np.array(kpts), dtype=torch.float32) #[batch_size,24,2]
    cls = torch.as_tensor(cls, dtype=torch.long) #[batch_size]
    conds = [torch.as_tensor(c, dtype=torch.long) for c in cond]

    state_labels_per_ex = []
    num_states_per_ex = [5, 4, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 3, 4, 3, 3, 4, 5, 4, 4, 3, 5, 3, 3, 5, 5, 3, 3, 3, 4, 5]
    for ex_idx, n_state in enumerate(num_states_per_ex):
        tmp = torch.zeros(size=(len(batch),n_state), dtype=torch.long)
        for i, (lbl, cond_vec) in enumerate(zip(cls,conds)):
            if lbl.item() == ex_idx:
                tmp[i] = cond_vec
        state_labels_per_ex.append(tmp)

    return img_paths, kpts, cls, state_labels_per_ex

def joints_to_bones(joints: torch.tensor) -> torch.tensor:
    N,C,T,V,M = joints.shape
    device = joints.device; dtype = joints.dtype

    parents = torch.tensor([p for p, _ in EDGE_IDX], device=device, dtype=torch.long)
    childs = torch.tensor([c for _, c in EDGE_IDX], device=device, dtype=torch.long)

    bones = torch.zeros_like(joints)
    bones[:,:,:,childs,:] = joints[:,:,:,childs,:] - joints[:,:,:,parents,:]
    return bones

def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vision_model = YOLO("ckpts/yolo11s_pose_best.pt")
    action_model_bone = MultiHeadAGCN(
        num_exercises=41,
        num_states_per_exercise=[5, 4, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 3, 4, 3, 3, 4, 5, 4, 4, 3, 5, 3, 3, 5, 5, 3, 3, 3, 4, 5],
        num_point = 24,
        num_person = 1,
        graph= "model_action.arch.graph.mygraph.Graph",
        in_channels=2,
        drop_out=0.5
    )
    action_model_joint = MultiHeadAGCN(
        num_exercises=41,
        num_states_per_exercise=[5, 4, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 3, 4, 3, 3, 4, 5, 4, 4, 3, 5, 3, 3, 5, 5, 3, 3, 3, 4, 5],
        num_point = 24,
        num_person = 1,
        graph= "model_action.arch.graph.mygraph.Graph",
        in_channels=2,
        drop_out=0.5
    )
    
    use_vision = False
    alpha = 0.5

    action_model_bone.to(device)
    action_model_joint.to(device)

    action_model_bone.eval()
    action_model_joint.eval()

    dataset = TestDataset("S:/FIT/Label2","S:/FIT/Img","cfg/exercise_condition_map.json")
    dataloader = DataLoader(dataset,batch_size=8,collate_fn=collate_fn)
    
    num_exercise = 41
    ensemble = True
    use_bone = True

    # class classfication metric
    # Accuracy
    ex_right_per_class = torch.zeros(num_exercise, dtype=torch.long)
    ex_total_per_class = torch.zeros(num_exercise, dtype=torch.long)
    # F1
    ex_true_list = []; ex_pred_list = [] # saves all exercise class result
    
    # condition regression metric
    # Exact-Match(Accuracy)
    state_EM_right = torch.zeros(num_exercise, dtype=torch.long)
    state_EMALL_total = torch.zeros(num_exercise, dtype=torch.long)
    state_EMCOND_total = torch.zeros(num_exercise, dtype=torch.long)
    # F1 
    state_F1_uncond_pred = [[] for _ in range(num_exercise)]
    state_F1_uncond_true = [[] for _ in range(num_exercise)]
    state_F1_cond_pred = [[] for _ in range(num_exercise)]
    state_F1_cond_true = [[] for _ in range(num_exercise)] 
    # mAP
    state_AP_uncond_prob = [[] for _ in range(num_exercise)]
    state_AP_uncond_true = [[] for _ in range(num_exercise)]
    state_AP_cond_prob = [[] for _ in range(num_exercise)]
    state_AP_cond_true = [[] for _ in range(num_exercise)]
    
    for (img_paths, kpts, ex_label, state_label_list) in tqdm(dataloader,desc="Testing..."):
        # img_path : str[list]
        # kpts: [B,T,V,C]
        # cls: [B,]
        # cond: len = 41, for each item, [B,num_state]
        kpts = kpts.to(device); ex_label = ex_label.to(device);
        state_label_list = [state_label.to(device) for state_label in state_label_list]

        batch_size = kpts.size(0)
        
        if use_vision:
            vision_result_list = vision_model(img_paths, verbose=False, save=False, show=False)
            vision_result = torch.cat([r.keypoints.xy for r in vision_result_list],dim = 0) # (B*T,V,C)
            vision_result = vision_result.view(batch_size, 16, *vision_result.shape[1:]) # (B,T,V,C)
            joints = vision_result.permute(0,3,1,2).unsqueeze(-1)
        else:
            joints = kpts.permute(0,3,1,2).unsqueeze(-1)

        bones = joints_to_bones(joints) # (B, C, T, V, M)

        with torch.no_grad():
            if ensemble:
                ex_logit_j, state_logits_j = action_model_joint(joints)
                ex_logit_b, state_logits_b = action_model_bone(bones)
            
                ex_prob_j = torch.softmax(ex_logit_j,dim=-1); ex_prob_b = torch.softmax(ex_logit_b,dim=-1)
                ex_prob = alpha*ex_prob_j + (1-alpha)*ex_prob_b #(B,num_class)
                ex_pred = ex_prob.argmax(dim = 1) #(B)
                
                state_prob_j = [torch.sigmoid(logit) for logit in state_logits_j]
                state_prob_b = [torch.sigmoid(logit) for logit in state_logits_b]
                state_prob_list = [alpha*j + (1-alpha)*b for j,b in zip(state_prob_j,state_prob_b)] # len = 41, for each, (B,num_state)
            else:
                ex_logit, state_logits = action_model_bone(bones) if use_bone else action_model_joint(joints)

                ex_prob = torch.softmax(ex_logit,dim = -1)
                ex_pred = ex_prob.argmax(dim = 1)

                state_prob_list = [torch.sigmoid(logit) for logit in state_logits]


            ex_total_per_class += torch.bincount(ex_label, minlength=num_exercise).cpu()
            ex_right_per_class += torch.bincount(ex_label[ex_pred == ex_label], minlength=num_exercise).cpu()
            ex_true_list.append(ex_label.cpu()) # true goes to true list
            ex_pred_list.append(ex_pred.cpu()) # preds goes to pred list

        for ex_idx,(state_prob,state_label) in enumerate(zip(state_prob_list,state_label_list)):
            mask = ex_label == ex_idx
            if not mask.any(): continue

            prob_mask = state_prob[mask]; label_mask = state_label[mask]; pred_mask = (prob_mask > 0.5).int()
            # print(pred_mask.shape)
            state_EMALL_total[ex_idx] += mask.sum().item()

            state_F1_uncond_pred[ex_idx].append(pred_mask.cpu())
            state_F1_uncond_true[ex_idx].append(label_mask.cpu())

            state_AP_uncond_prob[ex_idx].append(prob_mask.cpu())
            state_AP_uncond_true[ex_idx].append(label_mask.cpu())

            true_mask = mask & (ex_label == ex_pred)
            if true_mask.any():
                prob_mask = state_prob[mask]; label_mask = state_label[mask]; pred_mask = (prob_mask > 0.5).int()

                # Update F1 per-class
                state_F1_cond_pred[ex_idx].append(pred_mask.cpu())
                state_F1_cond_true[ex_idx].append(label_mask.cpu())

                state_AP_cond_prob[ex_idx].append(prob_mask.cpu())
                state_AP_cond_true[ex_idx].append(label_mask.cpu())

                # Update Exact-Match
                eq_true = (pred_mask == label_mask).all(dim = 1)
                state_EM_right[ex_idx] += eq_true.sum().item()
                state_EMCOND_total[ex_idx] += true_mask.sum().item()

    cls_true = torch.cat(ex_true_list).cpu().numpy() #(data_len, )
    cls_pred = torch.cat(ex_pred_list).cpu().numpy() #(data_len, )

    # Precision, Recall, F1 per class
    cls_pre_list, cls_rec_list, cls_f1_list, cls_support_list = precision_recall_fscore_support(
        cls_true,cls_pred,average=None,labels=range(num_exercise),zero_division=0
    )
    # Precision, Recall, F1 all class with macro
    cls_pre_macro, cls_rec_macro, cls_f1_macro, _ = precision_recall_fscore_support(
        cls_true,cls_pred,average="macro",labels=range(num_exercise),zero_division=0
    )
    # Precision, Recall, F1 all class with weighted
    cls_pre_weight, cls_rec_weight, cls_f1_weight, _ = precision_recall_fscore_support(
        cls_true,cls_pred,average='weighted',labels=range(num_exercise),zero_division=0
    )

    # Class Classification Accuracy(total is same with micro F1)
    cls_acc_list = ex_right_per_class.float() / ex_total_per_class.float()
    cls_acc_total = ex_right_per_class.sum().float() / ex_total_per_class.sum().float()
    
    # State Uncond F1
    uncond_f1_list = [] # per class
    uncond_support_list = []
    for e in range(num_exercise):
        if len(state_F1_uncond_true[e]) == 0:
            uncond_f1_list.append(0.0)
            uncond_support_list.append(0)
            continue
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
    for e in range(num_exercise):
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
    for e in range(num_exercise):
        if len(state_AP_uncond_true[e]) == 0:
            uncond_AP_list.append(0.0)
            continue
        y_prob_e = torch.cat(state_AP_uncond_prob[e],dim=0).numpy()
        y_true_e = torch.cat(state_AP_uncond_true[e],dim=0).numpy()
        uncond_AP_list.append(average_precision_score(y_true_e,y_prob_e,average='macro'))
    uncond_mAP = np.mean(uncond_AP_list)
    
    cond_AP_list = []
    for e in range(num_exercise):
        if len(state_AP_cond_true[e]) == 0:
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

    with open("inference_result.json","w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=4)
        

if __name__ == '__main__':
    # dataset = TestDataset("S:/FIT/Label2","S:/FIT/Img","cfg/exercise_mapping.json")
    # exit()
    # parser = argparse.ArgumentParser(description="usage: test.py --config your_cfg_yaml_file")
    # parser.add_argument("--config",type=str,required=False,help="Your yaml file path",default="cfg/config.yaml")
    # parser.add_argument("--mode",type=str,required=False,default="test")
    # args = parser.parse_args()
    # cfg = OmegaConf.load(args.config)
    
    main()