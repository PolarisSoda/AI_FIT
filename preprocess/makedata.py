import os
import json
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==============================
# 1. 운동별 조건 정의 불러오기
# ==============================
def load_label_def(label_json_path):
    with open(label_json_path, "r", encoding="utf-8") as f:
        ex_conditions = json.load(f)

    exercises = list(ex_conditions.keys())
    exercise_to_idx = {name: i for i, name in enumerate(exercises)}
    exercise_to_conditions = {name: conds for name, conds in ex_conditions.items()}

    return exercises, exercise_to_idx, exercise_to_conditions


# ==============================
# 2. 관절 / edge 정의
# ==============================
JOINTS = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]
joint_to_idx = {name: i for i, name in enumerate(JOINTS)}

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
edge_idx = [(joint_to_idx[p], joint_to_idx[c]) for p, c in EDGES]


# ==============================
# 3. JSON → Joint array 변환
# ==============================
def parse_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if len(frames) == 0:
        return [], data.get("type_info", {})

    T = len(frames)
    V = len(JOINTS)
    C = 2

    view_keys = list(frames[0].keys())  # ex: ["view1", "view2", ...]

    samples = []
    for view_key in view_keys:
        arr = np.zeros((C, T, V, 1), dtype=np.float32)  # M=1
        for t, frame in enumerate(frames):
            pts = frame[view_key].get("pts", {})
            for j, joint in enumerate(JOINTS):
                if joint in pts:
                    arr[0, t, j, 0] = pts[joint]["x"] / 1920
                    arr[1, t, j, 0] = pts[joint]["y"] / 1080
        samples.append(arr)

    return samples, data.get("type_info", {})


def joints_to_bones(joint_arr):
    C, T, V, M = joint_arr.shape
    bones = np.zeros_like(joint_arr)
    for (p, c) in edge_idx:
        bones[:, :, c, :] = joint_arr[:, :, c, :] - joint_arr[:, :, p, :]
    return bones


# ==============================
# 4. 라벨 생성
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


# ==============================
# 5. 데이터셋 빌드 + Split
# ==============================
def build_dataset_split(data_json_dir, label_json_path,
                        out_train="train.pkl", out_val="val.pkl",
                        test_size=0.2, seed=42):
    exercises, exercise_to_idx, exercise_to_conditions = load_label_def(label_json_path)

    samples = []
    files = [f for f in os.listdir(data_json_dir) if f.endswith(".json")]

    for fname in tqdm(files, desc="Processing JSONs"):
        if "3d" in fname: continue
        path = os.path.join(data_json_dir, fname)

        joint_list, type_info = parse_json(path)

        if len(joint_list) == 0:
            # print(f"⚠️ Skipped {fname}: no frames")
            continue
        elif joint_list[0].shape[1] != 16:
            # print(f"⚠️ Skipped {fname}: not enough frames")
            continue


        ex_label, cond_vec = make_labels(type_info, exercise_to_idx, exercise_to_conditions)

        for joint_arr in joint_list:  # view별로 독립 샘플
            bone_arr = joints_to_bones(joint_arr)
            samples.append((joint_arr, bone_arr, ex_label, cond_vec))

    # train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(samples)), test_size=test_size, random_state=seed, shuffle=True
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    meta = {
        "exercises": exercises,
        "exercise_to_idx": exercise_to_idx,
        "exercise_to_conditions": exercise_to_conditions,
    }

    with open(out_train, "wb") as f:
        pickle.dump({"samples": train_samples, "meta": meta}, f)
    with open(out_val, "wb") as f:
        pickle.dump({"samples": val_samples, "meta": meta}, f)

    print(f"✅ Train: {len(train_samples)} samples → {out_train}")
    print(f"✅ Val: {len(val_samples)} samples → {out_val}")


if __name__ == "__main__":
    build_dataset_split(
        data_json_dir=r"S:/AI_FIT/datasets/Label",             # JSON 데이터 폴더
        label_json_path=r"S:/AI_FIT/cfg/exercise_condition_map.json",  # 운동별 조건 정의 JSON
        out_train="datasets/train_data.pkl",
        out_val="datasets/val_data.pkl",
        test_size=0.2,
        seed=42
    )
