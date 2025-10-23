# Re-run the projection + plot (env was reset)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Skeleton spec
JOINTS = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]
J2I = {n:i for i,n in enumerate(JOINTS)}
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
EDGE_IDX = [(J2I[a], J2I[b]) for a,b in EDGES]

pts = {"Nose": {"x": 1.3490971326828003, "y": 130.51170349121094, "z": 43.80257797241211}, "Left Eye": {"x": 3.342142105102539, "y": 134.22955322265625, "z": 44.626190185546875}, "Right Eye": {"x": -0.912956953048706, "y": 134.4597930908203, "z": 44.56474685668945}, "Left Ear": {"x": 7.196595668792725, "y": 138.64466857910156, "z": 32.12103271484375}, "Right Ear": {"x": -6.757478713989258, "y": 138.3763885498047, "z": 33.12612533569336}, "Left Shoulder": {"x": 20.011756896972656, "y": 120.26763153076172, "z": 20.241174697875977}, "Right Shoulder": {"x": -16.550504684448242, "y": 121.60931396484375, "z": 18.36454200744629}, "Left Elbow": {"x": 28.229829788208008, "y": 95.07624053955078, "z": 12.624078750610352}, "Right Elbow": {"x": -24.38056755065918, "y": 95.48765563964844, "z": 14.174598693847656}, "Left Wrist": {"x": 28.532060623168945, "y": 71.18666076660156, "z": 24.099079132080078}, "Right Wrist": {"x": -25.71536636352539, "y": 71.5710678100586, "z": 22.479713439941406}, "Left Hip": {"x": 14.077181816101074, "y": 70.57412719726562, "z": 0.31200864911079407}, "Right Hip": {"x": -7.671466827392578, "y": 70.39898681640625, "z": -0.7370897531509399}, "Left Knee": {"x": 10.688045501708984, "y": 33.834556579589844, "z": -2.435192108154297}, "Right Knee": {"x": -4.589347839355469, "y": 34.44526290893555, "z": -2.403642177581787}, "Left Ankle": {"x": 7.435518741607666, "y": 1.5219208002090454, "z": -13.495014190673828}, "Right Ankle": {"x": -2.0244100093841553, "y": 2.2620151042938232, "z": -13.9898099899292}, "Neck": {"x": 0.9388756155967712, "y": 129.37220764160156, "z": 25.40570831298828}, "Left Palm": {"x": 29.607921600341797, "y": 66.83908081054688, "z": 27.687707901000977}, "Right Palm": {"x": -26.422922134399414, "y": 66.47303771972656, "z": 24.942806243896484}, "Back": {"x": 1.8570743799209595, "y": 105.11722564697266, "z": 13.962569236755371}, "Waist": {"x": 2.5082497596740723, "y": 86.82699584960938, "z": 6.735588550567627}, "Left Foot": {"x": 6.84525728225708, "y": -5.475880146026611, "z": -7.811266899108887}, "Right Foot": {"x": -2.3760387897491455, "y": -4.934129238128662, "z": -8.955107688903809}}

def limited_rotation_matrix(yaw_range=(-0, 0), pitch_range=(-0, 0), roll_range=(-0, 0)):
    yaw = np.deg2rad(np.random.uniform(*yaw_range))
    pitch = np.deg2rad(np.random.uniform(*pitch_range))
    roll = np.deg2rad(np.random.uniform(*roll_range))
    print(yaw,pitch,roll)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw),  np.cos(yaw), 0],[0, 0, 1]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],[0, 1, 0],[-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0],[0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll),  np.cos(roll)]])
    return Rz @ Ry @ Rx

def random_orthographic_projection(joint_3d, yaw_range=(-120,120), pitch_range=(-5,5), roll_range=(-20,20)):
    C,T,V,M = joint_3d.shape
    center = joint_3d.mean(axis=(1,2,3), keepdims=True)  # 회전 중심
    centered = joint_3d - center
    R = limited_rotation_matrix(yaw_range, pitch_range, roll_range)
    rotated = np.tensordot(R, centered, axes=([1],[0]))
    rotated += center
    proj = rotated[[0, 2], :, :, :]  # (x,z) 투영

    return proj, center  # 중심 함께 반환
# pack to (3,1,V,1)
V = len(JOINTS)
arr3d = np.zeros((3,1,V,1), dtype=np.float32)
for j,name in enumerate(JOINTS):
    p = pts[name]
    arr3d[0,0,j,0] = p["x"]
    arr3d[1,0,j,0] = p["y"]
    arr3d[2,0,j,0] = p["z"]

# projection 실행
proj, center = random_orthographic_projection(arr3d)
xy = proj[:,0,:,0]
cx, cz = center[0,0,0,0], center[2,0,0,0]  # 중심 (x,z)

# plot
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(xy[0], xy[1], label="Joints")
for i,j in EDGE_IDX:
    ax.plot([xy[0,i], xy[0,j]], [xy[1,i], xy[1,j]], color="gray", lw=1)
ax.scatter(cx, cz, color="red", s=80, label="Center (pivot)")

#ax.set_aspect('equal', 'box')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_title("2D orthographic projection (x,z) with rotation center")
ax.set_xlabel("X")
ax.set_ylabel("Z (inverted)")
ax.legend()

out_path = Path("pose2d_projection_center.png")
fig.savefig(out_path, bbox_inches="tight")
print(out_path.as_posix())
