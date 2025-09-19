# graph/mygraph.py
import numpy as np

JOINTS = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]
name2idx = {n:i for i,n in enumerate(JOINTS)}

# parent -> child (당신이 전처리 때 썼던 EDGES와 동일해야 합니다)
EDGES_NAME = [
    ("Left Shoulder", "Left Elbow"),
    ("Left Elbow", "Left Wrist"),
    ("Left Wrist", "Left Palm"),
    ("Right Shoulder", "Right Elbow"),
    ("Right Elbow", "Right Wrist"),
    ("Right Wrist", "Right Palm"),
    ("Waist", "Left Hip"),
    ("Left Hip", "Left Knee"),
    ("Left Knee", "Left Ankle"),
    ("Left Ankle", "Left Foot"),
    ("Waist", "Right Hip"),
    ("Right Hip", "Right Knee"),
    ("Right Knee", "Right Ankle"),
    ("Right Ankle", "Right Foot"),
    ("Neck", "Back"),
    ("Back", "Waist"),
    ("Neck", "Left Shoulder"),
    ("Neck", "Right Shoulder"),
    ("Neck", "Nose"),
    ("Nose", "Left Eye"),
    ("Nose", "Right Eye"),
    ("Left Eye", "Left Ear"),
    ("Right Eye", "Right Ear"),
]
EDGES = [(name2idx[p], name2idx[c]) for p,c in EDGES_NAME]

def edge2mat(edges, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edges:
        A[j, i] = 1.0  # i -> j (inward에서 j가 타겟)
    return A

def normalize_digraph(A):
    Dl = np.sum(A, axis=0)  # column sum
    Dl[Dl == 0] = 1
    Dn = np.diag(1.0 / Dl)
    return A @ Dn

class Graph:
    """
    ST-GCN 스타일의 3분기 인접행렬 (self, inward, outward).
    num_node=24 고정 (위 JOINTS와 일치)
    """
    def __init__(self, labeling_mode='spatial'):
        num_node = len(JOINTS)  # 24
        self_link = [(i, i) for i in range(num_node)]
        inward = [(c, p) for (p, c) in EDGES]   # (dst, src)
        outward = [(p, c) for (p, c) in EDGES]  # (dst, src)

        I = edge2mat(self_link, num_node)
        In = edge2mat(inward, num_node)
        Out = edge2mat(outward, num_node)

        I = normalize_digraph(I)
        In = normalize_digraph(In)
        Out = normalize_digraph(Out)

        self.A = np.stack((I, In, Out)).astype(np.float32)  # (3, 24, 24)
