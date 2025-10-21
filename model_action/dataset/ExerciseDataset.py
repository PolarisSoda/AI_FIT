import numpy as np
import pickle
import random
import torch

from torch.utils.data import Dataset

def limited_rotation_matrix(yaw_range=(-60, 60), pitch_range=(-30, 30), roll_range=(-10, 10)):
    yaw = np.deg2rad(np.random.uniform(*yaw_range))
    pitch = np.deg2rad(np.random.uniform(*pitch_range))
    roll = np.deg2rad(np.random.uniform(*roll_range))

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    return Rz @ Ry @ Rx

def random_orthographic_projection(
    joint_3d,
    yaw_range=(-20,20),
    pitch_range=(-20,20), 
    roll_range=(-3,3)
):
    C,T,V,M = joint_3d.shape

    # 중심: (3,1,1,1)로 만들어 브로드캐스트
    center = joint_3d.mean(axis=(1,2,3), keepdims=True)   # (3,1,1,1)
    centered = joint_3d - center                          # (3,T,V,M)

    # 회전행렬
    R = limited_rotation_matrix(yaw_range, pitch_range, roll_range)  # (3,3)
    # tensordot으로 (3,3)·(3,T,V,M) -> (3,T,V,M)
    rotated = np.tensordot(R, centered, axes=([1],[0]))      # (3,T,V,M)
    rotated += center

    proj = rotated[:2]                                       # (2,T,V,M)
    proj[1] *= -1

    # 최소한의 보정 같은느낌적인느낌느낌
    x = proj[0]
    y = proj[1]
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    width  = float(maxx - minx)
    height = float(maxy - miny)
    long_side = max(width, height)
    if long_side < 1e-8:
        scale = 1.0
    else:
        scale = max(1.0, 540.0 / long_side)
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    proj[0] = (x - cx) * scale + cx
    proj[1] = (y - cy) * scale + cy

    # 왠진 모르겠지만 이걸 하면 또 loss가 확 괜찮아짐.
    # W, H, pad = 1920, 1080, 2

    # x = proj[0]; y = proj[1]
    # minx, maxx = float(x.min()), float(x.max())
    # miny, maxy = float(y.min()), float(y.max())
    # cx = 0.5 * (minx + maxx); cy = 0.5 * (miny + maxy)
    # width  = max(maxx - minx, 1e-6)
    # height = max(maxy - miny, 1e-6)

    # # 1) 프레임 안에 들어오게 하는 최대 스케일
    # s_max_x = max((W - 2*pad) / width,  0.0)
    # s_max_y = max((H - 2*pad) / height, 0.0)
    # s_rand  = np.random.uniform(0.9, 1.1)   # 증강 강도는 여기서 조절
    # s = min(s_rand, s_max_x, s_max_y)

    # # 2) 먼저 스케일만 적용
    # x0 = (x - cx) * s + cx
    # y0 = (y - cy) * s + cy

    # proj[0] = x0
    # proj[1] = y0
    return proj

def safe_scale(joint_2d,W=1920,H=1080,pad=2):
    x = joint_2d[0]
    y = joint_2d[1]

    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    width  = maxx - minx
    height = maxy - miny

    # 폭/높이가 0인 특수 케이스 방어
    if width == 0:
        width = 1e-6
    if height == 0:
        height = 1e-6

    half_w = width  * 0.5
    half_h = height * 0.5

    # 센터 기준 여유 거리 (양쪽 중 작은 쪽이 제한이 됨)
    margin_left   = cx - 0.0
    margin_right  = W - cx
    margin_top    = cy - 0.0
    margin_bottom = H - cy

    allow_half_w = max(min(margin_left, margin_right) - pad, 0.0)
    allow_half_h = max(min(margin_top,  margin_bottom) - pad, 0.0)

    smax_x = allow_half_w / half_w if half_w > 0 else np.inf
    smax_y = allow_half_h / half_h if half_h > 0 else np.inf

    s_rand = np.random.uniform(0.5, 1.5)
    s_safe = min(s_rand, smax_x, smax_y)
    s_safe = max(safe_small := 0.0, s_safe)

    joint_2d[0] = (x - cx) * s_safe + cx
    joint_2d[1] = (y - cy) * s_safe + cy

    return joint_2d

def boundary_box_centering(joint_2d,normalize):
    minx = np.min(joint_2d[0,:,:,:]); maxx = np.max(joint_2d[0,:,:,:])
    miny = np.min(joint_2d[1,:,:,:]); maxy = np.max(joint_2d[1,:,:,:])

    cx = (minx+maxx) / 2 ; cy = (miny+maxy) / 2;
    joint_2d[0] -= cx; joint_2d[1] -= cy;

    if normalize == 'unit': # 0 ~ 1 normalization
        joint_2d[0] = (joint_2d[0] + 960) / 1920
        joint_2d[1] = (joint_2d[1] + 540) / 1080
    else: # -1 ~ 1 normalization
        joint_2d[0] /= 960
        joint_2d[1] /= 540
    
    return joint_2d

def joints_to_bones(joint_arr):
    parents = np.array([5,7,9,6,8,10,20,11,13,15,20,12,14,16,17,19,17,17,17,0,0,1,2])
    children = np.array([7,9,17,8,10,18,11,13,15,21,12,14,16,22,19,20,5,6,0,1,2,3,4])

    bones = np.zeros_like(joint_arr)
    bones[:, :, children, :] = joint_arr[:, :, children, :] - joint_arr[:, :, parents, :]
    return bones

class ExerciseDataset(Dataset):
    def __init__(self, 
        pickle_path, 
        use_bone=False, 
        aug_prob=0.5,
        normalize='-1~1', 
        center_translate=True, 
        scale_aug_prob=0.9, 
        use_3d_aug=True,
        is_val=False,
        val_scale_range=(0.9, 1.5),
        val_base_seed=42,
        **kwargs):

        data = pickle.load(open(pickle_path, "rb"))
        self.samples = data["samples"]
        self.meta = data["meta"]

        self.use_bone = use_bone
        self.aug_prob = aug_prob
        self.normalize = normalize
        self.center_translate = center_translate
        self.scale_aug_prob = scale_aug_prob
        self.use_3d_aug=use_3d_aug

        if is_val:
            print("This is Validation set! Pre-applying 2D center-scale-translate once (fit inside image).")
            rng = np.random.default_rng(val_base_seed)
            W = 1920; H = 1080
            pad = 2
            new_samples = []
            for i, sample in enumerate(self.samples):
                joint_2d, second, joint_3d, ex_label, cond_label = sample
                j2 = np.asarray(joint_2d, dtype=np.float32).copy()

                x = j2[0]; y = j2[1]
                minx, maxx = float(x.min()), float(x.max())
                miny, maxy = float(y.min()), float(y.max())
                cx = 0.5 * (minx + maxx)
                cy = 0.5 * (miny + maxy)

                width  = max(maxx - minx, 1e-6)
                height = max(maxy - miny, 1e-6)

                # 1) 랜덤 스케일 뽑되, 화면 안에 들어올 수 있도록 상한 클램프
                s_rand = float(rng.uniform(*val_scale_range))
                s_max_x = max((W - 2*pad) / width,  0.0)
                s_max_y = max((H - 2*pad) / height, 0.0)
                s = min(s_rand, s_max_x, s_max_y)

                # 2) 스케일만 먼저 적용(translation 없이)
                x0 = (x - cx) * s + cx
                y0 = (y - cy) * s + cy

                # 3) "모든 키포인트가 이미지 내부"가 되도록 tx, ty의 허용구간 계산
                #    tx_min <= tx <= tx_max,  ty_min <= ty <= ty_max
                tx_min = pad - float(x0.min())
                tx_max = (W - pad) - float(x0.max())
                ty_min = pad - float(y0.min())
                ty_max = (H - pad) - float(y0.max())

                # 안전상 소수 오차 방어
                if tx_min > tx_max: 
                    # 이 경우는 s 클램프가 약간 부족한 수치오차 상황일 수 있음 -> 중앙 배치
                    tx = 0.5 * (tx_min + tx_max)
                else:
                    tx = float(rng.uniform(tx_min, tx_max))

                if ty_min > ty_max:
                    ty = 0.5 * (ty_min + ty_max)
                else:
                    ty = float(rng.uniform(ty_min, ty_max))

                # 4) 최종 적용: (center-scale) 후 허용구간 안에서 랜덤 translation
                j2[0] = x0 + tx
                j2[1] = y0 + ty

                # (선택) 혹시 모를 경계 초과 미세오차 클립
                # j2[0] = np.clip(j2[0], pad, W - pad)
                # j2[1] = np.clip(j2[1], pad, H - pad)

                new_samples.append((j2, second, joint_3d, ex_label, cond_label))

            self.samples = new_samples
        else:
            print(f"[DEBUG] Centring + Random Scale : {self.center_translate}, 3D->2D Projection: {self.use_3d_aug} for Training")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        joint_2d, _, joint_3d, ex_label, cond_label = self.samples[idx]
        joint_2d = np.array(joint_2d, dtype=np.float32, copy=True)
        joint_3d = np.array(joint_3d, dtype=np.float32, copy=True)
        
        # No Augmentation goes to direct return
        if self.center_translate == True:
            if not self.is_val:
                # Whether Use 3d->2d Projection and probability accepted
                if self.use_3d_aug and np.random.random() < self.aug_prob:
                    joint_2d = random_orthographic_projection(joint_3d)

                # In this Section you should do random scale with probability of scale_aug_prob
                if np.random.random() < self.scale_aug_prob:
                    joint_2d = safe_scale(joint_2d)

            joint_2d = boundary_box_centering(joint_2d,self.normalize)


        if self.use_bone: joint_2d = joints_to_bones(joint_2d)

        x = torch.from_numpy(joint_2d).float()
        ex_label = torch.tensor(ex_label, dtype=torch.long)
        cond_label = torch.from_numpy(cond_label).float()

        return x,ex_label,cond_label
