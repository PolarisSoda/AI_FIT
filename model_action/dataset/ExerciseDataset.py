import torch
from torch.utils.data import Dataset
import pickle


class ExerciseDataset(Dataset):
    def __init__(self, pkl_path, use_bone=False, **kwargs):
        """
        pkl_path : str, pickle 파일 경로
        use_bone : bool, True면 bone_arr, False면 joint_arr 사용
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.samples = data["samples"]  # [(joint_arr, bone_arr, ex_label, cond_vec), ...]
        self.meta = data["meta"]
        self.use_bone = use_bone

        # 운동별 상태 개수 (멀티헤드 학습용)
        self.num_states_per_ex = [len(self.meta["exercise_to_conditions"][ex])
                                  for ex in self.meta["exercises"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        joint_arr, bone_arr, ex_label, cond_vec = self.samples[idx]

        # numpy → torch
        if self.use_bone:
            x = torch.from_numpy(bone_arr).float()   # [2, T, 24, 1]
        else:
            x = torch.from_numpy(joint_arr).float()  # [2, T, 24, 1]

        ex_label = torch.tensor(ex_label, dtype=torch.long)
        cond_vec = torch.from_numpy(cond_vec).float()

        return x, ex_label, cond_vec
