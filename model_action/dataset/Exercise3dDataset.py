import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class Exercise3dDataset(Dataset):
    def __init__(self,
        pkl_path: str, 
        use_bone: bool = False,
        augment_prob: float = 0.5,
        cam_cfg : dict | None = None,
        bone_from_joint = None,
        bone_edges: list[tuple[int,int]] | None = None,
        seed: int = 42,
    ):
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.samples = data["samples"]  # (joint_2d,bone_2d,joint_3d,bone_3d,ex_label,cond_vec)
        self.meta = data["meta"]
        self.use_bone = use_bone

        # 운동별 상태 개수 (멀티헤드 학습용)
        self.num_states_per_ex = [len(self.meta["exercise_to_conditions"][ex]) for ex in self.meta["exercises"]]

        self.augment_prob = augment_prob
        self.rng = np.random.default_rng(seed)
        self.cam_cfg = cam_cfg or {}
        self.bone_from_joint = bone_from_joint
        self.bone_edges = bone_edges


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        joint_2d,bone_2d,joint_3d,bone_3d,ex_label,cond_vec = self.samples[idx]
        
        # numpy → torch
        if self.use_bone:
            x = torch.from_numpy(bone_arr).float()   # [2, T, 24, 1]
        else:
            x = torch.from_numpy(joint_arr).float()  # [2, T, 24, 1]

        ex_label = torch.tensor(ex_label, dtype=torch.long)
        cond_vec = torch.from_numpy(cond_vec).float()

        return x, ex_label, cond_vec
