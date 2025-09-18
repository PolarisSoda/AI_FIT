import os
import json
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Define joint, edges and their indices
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
EDGE_IDX = [(JOINT_IDX[p],JOINT_IDX[c]) for p,c in EDGES]

VIEWS = ["view1","view2","view3","view4","view5"]

# load exercises and their conditions
def load_label_def(label_json_path: str):
    with open(label_json_path,"r",encoding="utf-8") as f:
        ex_conditions = json.load(f)

    exercises = list(ex_conditions.keys()) # exercise list
    exercise_to_idx = {name: i for i,name in enumerate(exercises)} # and we mapped each idx to exercise, e.g) "buffy test" : 2
    exercise_to_conditions = {name: conds for name,conds in ex_conditions.items()} # each exercise has own conditions
    return exercises, exercise_to_idx, exercise_to_conditions

# ==============================
# 3. 유틸: JSON 안전 파서
# ==============================
def _frame_iter(frames_obj):
    """frames가 list 또는 dict여도 통일된 순서로 순회"""
    if isinstance(frames_obj, list):
        for i, fr in enumerate(frames_obj):
            yield str(i), fr
    elif isinstance(frames_obj, dict):
        # 숫자 키/문자 키 혼용 지원
        def _order(k):
            try: return int(k)
            except: return k
        for k in sorted(frames_obj.keys(), key=_order):
            yield k, frames_obj[k]
    else:
        return

def _get_xy(pt):
    """2D 포인트: dict{'x','y'} or list/tuple[2] 모두 지원"""
    if pt is None: return None
    if isinstance(pt, dict):
        if 'x' in pt and 'y' in pt:
            return float(pt['x']), float(pt['y'])
        # 혹시 다른 키로 온다면 최대한 추정
        ks = list(pt.keys())
        if len(ks) >= 2:
            a, b = pt[ks[0]], pt[ks[1]]
            try: return float(a), float(b)
            except: return None
    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    return None

def _get_xyz(pt):
    """3D 포인트: dict{'x','y','z'} or list/tuple[3] 지원"""
    if pt is None: return None
    if isinstance(pt, dict):
        if all(k in pt for k in ('x','y','z')):
            return float(pt['x']), float(pt['y']), float(pt['z'])
        ks = list(pt.keys())
        if len(ks) >= 3:
            try: return float(pt[ks[0]]), float(pt[ks[1]]), float(pt[ks[2]])
            except: return None
    elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
        return float(pt[0]), float(pt[1]), float(pt[2])
    return None

def _get_joint_case_insensitive(pts_dict, joint_name):
    """조인트 이름 대소문자/공백 차이 방어"""
    if joint_name in pts_dict:
        return pts_dict[joint_name]
    low = joint_name.lower()
    for k in pts_dict.keys():
        if k.lower() == low:
            return pts_dict[k]
    return None


# ==============================
# 4. JSON → 2D/3D 배열 변환
# ==============================
def parse_json_2d(json_path: str):
    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    T = len(frames); V = len(JOINTS); C = 2
    samples = []

    for view in VIEWS:
        arr = np.zeros(shape=(C,T,V,1),dtype=np.float32)
        for i,frame in enumerate(frames):
            pts = frame[view]["pts"]
            for j,joint in enumerate(JOINTS):
                arr[0,i,j,0] = pts[joint]["x"]
                arr[1,i,j,0] = pts[joint]["y"]
        samples.append(arr)

    return samples, data["type_info"]

def parse_json_3d(json_path: str):
    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    T = len(frames); V = len(JOINTS); C = 3

    arr = np.zeros(shape=(C,T,V,1),dtype=np.float32)
    for i,frame in enumerate(frames):
        pts = frame["pts"]
        for j,joint in enumerate(JOINTS):
            arr[0,i,j,0] = pts[joint]["x"]
            arr[1,i,j,0] = pts[joint]["y"]
            arr[2,i,j,0] = pts[joint]["z"]

    return arr


def joints_to_bones(joint_arr):
    bones = np.zeros_like(joint_arr)
    for p,c in EDGE_IDX:
        bones[:,:,c,:] = joint_arr[:, :, c, :] - joint_arr[:, :, p, :]
    return bones


# ==============================
# 5. 라벨 생성
# ==============================
def make_labels(type_info, exercise_to_idx, exercise_to_conditions):
    ex_name = type_info["exercise"]
    ex_idx = exercise_to_idx[ex_name]
    cond_list = exercise_to_conditions[ex_name]

    cond_vec = np.zeros(len(cond_list), dtype=np.float32)
    for cond in type_info["conditions"]:
        name, val = cond["condition"], cond["value"]
        if name in cond_list:
            idx = cond_list.index(name)
            cond_vec[idx] = float(val)

    return ex_idx, cond_vec


def build_dataset_split(data_json_dir: str, label_json_path: str, out_train: str, out_val: str, test_size: float, seed: int):
    exercises, exercise_to_idx, exercise_to_conditions = load_label_def(label_json_path)

    samples = []
    
    files = []
    files_2d = [file for file in os.listdir(data_json_dir) if not "-3d" in file]
    
    for file in files_2d:
        prefix = file.split(".")[0]
        file_path = os.path.join(data_json_dir,f"{prefix}-3d.json")
        if os.path.exists(file_path): files.append(prefix)

    for file_name in tqdm(files, desc="Processing JSON...:"):
        path_2d = os.path.join(data_json_dir, f"{file_name}.json")
        path_3d = os.path.join(data_json_dir, f"{file_name}-3d.json")

        joint_list_2d, type_info = parse_json_2d(path_2d)
        joint_3d = parse_json_3d(path_3d)
        bone_3d = joints_to_bones(joint_3d)

        if len(joint_list_2d) == 0: continue
        elif joint_list_2d[0].shape[1] != 16: continue

        ex_label,cond_vec = make_labels(type_info,exercise_to_idx,exercise_to_conditions)

        for joint_2d in joint_list_2d:
            bone_2d = joints_to_bones(joint_2d)
            samples.append((joint_2d,bone_2d,joint_3d,bone_3d,ex_label,cond_vec))
    
    train_idx,val_idx = train_test_split(
        np.arange(len(samples)),test_size=test_size,random_state=seed,shuffle=True
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    meta = {
        "exercises": exercises,
        "exercise_to_idx": exercise_to_idx,
        "exercise_to_conditions": exercise_to_conditions,
    }

    with open(out_train,"wb") as f:
        pickle.dump({"samples": train_samples, "meta": meta}, f)
    with open(out_val, "wb") as f:
        pickle.dump({"samples": val_samples, "meta": meta}, f)

    print(f"Train: {len(train_samples)} samples: {out_train}")
    print(f"Val: {len(val_samples)} samples: {out_val}")


if __name__ == "__main__":
    build_dataset_split(
        data_json_dir="S:/AI_FIT/datasets/Label",
        label_json_path="S:/AI_FIT/cfg/exercise_condition_map.json",
        out_train="datasets/train_data_3d.pkl",
        out_val="datasets/val_data_3d.pkl",
        test_size=0.2,
        seed=42
    )
