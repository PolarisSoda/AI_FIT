import argparse
import os
import json
from torch.utils.data import Dataset,DataLoader
from omegaconf import OmegaConf, DictConfig
from ultralytics import YOLO
from model_action.arch import MultiHeadAGCN,TCNNetwork
from model_action.arch import TCNNetwork
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, hamming_loss
import shutil
import cv2
import numpy as np
import torch

ORDER = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]

def pts_to_array(pts: dict):
    out = []
    for k in ORDER:
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
        self.index = []

        for file in tqdm(self.file_list, desc="Initializing Datasets..."):
            with open(os.path.join(self.file_dir,file),"r",encoding="utf-8") as f:
                data = json.load(f)
            
            img_path, kpts = [], []
            cls = self.mapper[data["type_info"]["exercise"]]
            cond = [item["value"] for item in data["type_info"]["conditions"]]

            frames = data["frames"]
            for frame in data["frames"]:
                img_path.append(os.path.join(self.img_dir,frame["img_key"]))
                kpts.append(pts_to_array(frame.get("pts", {})))  # (24,2)

            self.index.append({
                "img_path": img_path,
                "kpts": kpts,
                "cls" : cls,
                "cond" : cond
            })

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        it = self.index[idx]
        return it["img_path"], it["kpts"], it["cls"], it["cond"]

def to_path_list(x):
    """
    x: list[str] 또는 list[(str,)] 같은 구조를 안전하게 list[str]로 변환
    """
    out = []
    for i, v in enumerate(x):
        if isinstance(v, (tuple, list)):
            if len(v) != 1 or not isinstance(v[0], str):
                raise TypeError(f"[to_path_list] [{i}] 예상치 못한 항목: {type(v)} -> {v}")
            out.append(v[0])
        elif isinstance(v, str):
            out.append(v)
        else:
            raise TypeError(f"[to_path_list] [{i}] 지원하지 않는 타입: {type(v)}")
    return out

def collate_bs1(batch):
    """
    batch_size=1 전제. DataLoader가 튜플을 한 겹 더 싸는 걸 방지하고
    곧바로 (seq_paths:list[str], kpts:np.ndarray(T,24,2), cls:int, cond:list)로 반환.
    """
    assert len(batch) == 1
    img_paths, kpts, cls, cond = batch[0]
    # 혹시라도 (path,) 튜플이 섞여 있으면 정리
    if any(isinstance(p, (tuple, list)) for p in img_paths):
        img_paths = to_path_list(img_paths)
    return img_paths, kpts, cls, cond



def main(cfg: DictConfig) -> None:
    # Vision Model Define
    if cfg.vision_type == 'yolo':
        pretrain = cfg.vision_pretrain_path if cfg.vision_pretrain_path else "yolo11s-pose.pt"
        vision_model = YOLO(pretrain)
    else:
        raise NotImplementedError("Others Not Implemented Yet")
    
    # Action Model Define
    if cfg.action_type == 'agcn':
        action_model = MultiHeadAGCN()
        if cfg.action_pretrain_path: action_model.load_pretrain(cfg.action_pretrain_path)
    elif cfg.action_type == 'tcn':
        action_model = TCNNetwork().to('cuda')
    elif cfg.action_type == 'sttformer':
        action_model = 1
    else:
        raise NotImplementedError("Others Not Implemented Yet")
    
    action_model.eval()
    dataset = TestDataset("S:/FIT/Label2","S:/FIT/Img","cfg/exercise_mapping.json")
    dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate_bs1)
    
    cls_correct = 0
    cls_total = 0
    y_true, y_pred = [], []

    for idx, (img_paths, kpts, cls, cond) in enumerate(tqdm(dataloader,desc="Testing...")):
        # YOLO 추론
        results = vision_model(img_paths, verbose=False, save=False, show=False)
        ckpt_output = torch.cat([r.keypoints.xy for r in results], dim=0)
        ckpt_output = ckpt_output.unsqueeze(0).flatten(2,3)
        
        with torch.no_grad():
            cls_output, cond_output = action_model(ckpt_output)

        cls_pred = cls_output.argmax(dim = -1).item()
        cls_total += 1
        cls_correct += int(cls_pred == cls)

        if cls_pred == cls:
            cond_probs = torch.sigmoid(cond_output[cls_pred])
            cond_pred = (cond_probs > 0.5).int()

            cond = torch.tensor(cond, dtype=torch.int, device=cond_pred.device)

            y_true.append(cond.detach().cpu().numpy().astype(np.int32))
            y_pred.append(cond_pred.detach().cpu().numpy().astype(np.int32))

        if cls_total == 100: break

        



        # # 매 5번째 시퀀스마다 시각화 저장
        # if idx % 5 == 0:
        #     for i, r in enumerate(results):
        #         orig_path = img_paths[i]               # 원본 프레임 경로
        #         fname = os.path.basename(orig_path)    # 파일명만 추출
        #         save_path = os.path.join("output", f"{idx:04d}_{fname}")

        #         # 원본 복사 후 YOLO 키포인트 덮어쓰기
        #         shutil.copy(orig_path, save_path)
        #         drawn = r.plot()
        #         cv2.imwrite(save_path, drawn)

    cls_acc = cls_correct / cls_total if cls_total else 0.0
    print(f"\n[Class] Accuracy: {cls_acc*100:.2f}%  ({cls_correct}/{cls_total})")
    if len(y_true) == 0:
        print("[Cond] 클래스가 맞은 샘플이 없어 상태 평가는 스킵되었습니다.")
    else:
        y_true = np.vstack(y_true)   # (N, C)
        y_pred = np.vstack(y_pred)   # (N, C)

        f1_micro  = f1_score(y_true, y_pred, average="micro",   zero_division=0)
        f1_macro  = f1_score(y_true, y_pred, average="macro",   zero_division=0)
        f1_weight = f1_score(y_true, y_pred, average="weighted",zero_division=0)

        subset_acc = accuracy_score(y_true, y_pred)      # 모든 라벨 완전일치(엄격)
        label_acc  = 1.0 - hamming_loss(y_true, y_pred) # 라벨별 평균 일치율(느슨)

        print(f"[Cond] Micro-F1 : {f1_micro*100:.2f}%")
        print(f"[Cond] Macro-F1 : {f1_macro*100:.2f}%")
        print(f"[Cond] Weighted-F1 : {f1_weight*100:.2f}%")
        print(f"[Cond] Subset Acc (all-match): {subset_acc*100:.2f}%")
        print(f"[Cond] Label Acc (mean per label): {label_acc*100:.2f}%")

        # 필요시 라벨별 리포트
        # print("\n[Cond] Per-label report:\n", classification_report(y_true, y_pred, zero_division=0))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="usage: test.py --config your_cfg_yaml_file")
    parser.add_argument("--config",type=str,required=False,help="Your yaml file path",default="cfg/config.yaml")
    parser.add_argument("--mode",type=str,required=False,default="test")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    main(cfg)
