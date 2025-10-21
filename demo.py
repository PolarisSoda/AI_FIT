# pip install gradio pillow opencv-python
import gradio as gr
from PIL import Image
import cv2
import os, time, math, re
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from model_action.arch import MultiHeadAGCN

# =========================
# 공통 설정/스켈레톤 정의
# =========================
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
NUM_POINT = 24
NUM_PERSON = 1
num_states = [5, 4, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 3, 4, 3, 3, 4, 5, 4, 4, 3, 5, 3, 3, 5, 5, 3, 3, 3, 4, 5]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 캐시된 본 벡터 인덱스
PARENTS_CPU = torch.tensor([p for p, _ in EDGE_IDX], dtype=torch.long)
CHILDS_CPU  = torch.tensor([c for _, c in EDGE_IDX], dtype=torch.long)

def joints_to_bones(joints: torch.Tensor) -> torch.Tensor:
    # joints: [N,C,T,V,M]
    parents = PARENTS_CPU.to(joints.device)
    childs  = CHILDS_CPU.to(joints.device)
    bones = torch.zeros_like(joints)
    bones[:, :, :, childs, :] = joints[:, :, :, childs, :] - joints[:, :, :, parents, :]
    return bones

vision_model = YOLO("ckpts/yolo11s_pose_best.pt")   # <- 네 체크포인트
action_model_bone = MultiHeadAGCN(
    num_exercises=41, num_states_per_exercise=num_states,
    num_point=NUM_POINT, num_person=NUM_PERSON,
    graph="model_action.arch.graph.mygraph.Graph",
    in_channels=2, drop_out=0.5
)
action_model_joint = MultiHeadAGCN(
    num_exercises=41, num_states_per_exercise=num_states,
    num_point=NUM_POINT, num_person=NUM_PERSON,
    graph="model_action.arch.graph.mygraph.Graph",
    in_channels=2, drop_out=0.5
)

vision_model.to(device)
action_model_bone.to(device)
action_model_joint.to(device)
action_model_bone.eval()
action_model_joint.eval()

# =========================
# 공통: 포즈 → [C,V] 키포인트 만들기
# =========================
def frame_to_keypoints_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """
    frame_bgr -> (2, V)  (x,y만 사용)
    YOLO keypoints.xy: [num_person, num_kpt, 2]
    """
    # YOLO 추론
    pred_device = 0 if torch.cuda.is_available() else 'cpu'
    results = vision_model.predict(source=frame_bgr, verbose=False, save=False, show=False, max_det=1, device=pred_device)
    result = results[0]
    kps = None
    try:
        xy = result.keypoints.xy
        if xy is not None and len(xy) > 0:
            arr = xy[0].detach().float().cpu().numpy()  # (24, 2)
            if arr.shape[0] >= NUM_POINT:
                kps = arr[:NUM_POINT, :]  # (24,2)
    except Exception:
        kps = None

    # ====== (데모용) 실패시 랜덤 fallback ======
    if kps is None:
        # 실제 서비스에서는: 이전 유효 포즈 유지(hold-last) 또는 프레임 스킵 권장
        kps = np.random.rand(NUM_POINT, 2).astype(np.float32)

    # 정규화(선택): 학습 스케일과 맞추세요. 여기선 [0,1]로
    H, W = frame_bgr.shape[:2]
    if W > 0 and H > 0:
        kps[:, 0] = kps[:, 0] / float(W)
        kps[:, 1] = kps[:, 1] / float(H)

    # (J,2) -> (2,J)
    return kps.T.astype(np.float32)  # (2, V)

def pack_for_agcn(seq_cv: list):
    """
    seq_cv: list of (2,V) 길이 16
    return: joints,bones as torch tensors: [1, 2, 16, V, 1]
    """
    x = np.stack(seq_cv, axis=1).astype(np.float32)  # (2,16,V)
    joints = torch.from_numpy(x)[None, :, :, :, None]  # [1,2,16,V,1]
    bones  = joints_to_bones(joints)
    return joints.to(device), bones.to(device)

@torch.inference_mode()
def agcn_forward(joints: torch.Tensor, bones: torch.Tensor):
    # 두 스트림 forward & 평균 ensemble
    ex_logit_j, state_logits_j = action_model_joint(joints)
    ex_logit_b, state_logits_b = action_model_bone(bones)
    ex_prob   = 0.5 * torch.softmax(ex_logit_j, dim=-1) + 0.5 * torch.softmax(ex_logit_b, dim=-1)
    ex_pred   = ex_prob.argmax(dim=1)               # [B]

    # 상태 확률(운동별 head)
    state_prob_j = [torch.sigmoid(logit) for logit in state_logits_j]
    state_prob_b = [torch.sigmoid(logit) for logit in state_logits_b]
    state_prob_list = [0.5 * j + 0.5 * b for j, b in zip(state_prob_j, state_prob_b)]
    return ex_pred, ex_prob, state_prob_list

# =========================
# (A) 업로드 영상: 0.25s 오프셋 스캔 (기존 유지)
# =========================
def read_frame_at_time(cap, fps, t_sec):
    idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return f if ok else None

def scan_video_offsets(video_path, offset_step=0.25, window_sec=16.0, sample_hz=1.0, conf_floor=0.0):
    if video_path is None or not os.path.exists(video_path):
        return None, "❌ 영상 파일이 없습니다.", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ 영상 열기 실패", None

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfrm = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur  = nfrm / fps if fps else 0.0
    if dur < window_sec:
        cap.release()
        return None, f"❌ 영상이 너무 짧습니다. 길이 {dur:.2f}s < {window_sec:.0f}s", None

    dt = 1.0 / sample_hz  # 1.0s
    max_start = dur - window_sec
    n_steps = int(math.floor(max_start / offset_step)) + 1

    best = {"conf": -1.0}
    all_scores = []

    for s_idx in range(n_steps):
        start = round(s_idx * offset_step, 3)
        times = [start + k * dt for k in range(int(window_sec / dt))]  # 16개

        seq = []
        valid = True
        for t in times:
            f = read_frame_at_time(cap, fps, t)
            if f is None:
                valid = False
                break
            kps = frame_to_keypoints_bgr(f)   # (2,V)
            seq.append(kps)
        if not valid or len(seq) != 16:
            all_scores.append(0.0)
            continue

        joints, bones = pack_for_agcn(seq)
        ex_pred, ex_prob, _ = agcn_forward(joints, bones)
        label = int(ex_pred[0].item())
        probs = ex_prob[0].detach().float().cpu().numpy()
        conf  = float(probs[label])
        all_scores.append(conf)

        if conf > best["conf"] and conf >= conf_floor:
            best = {
                "start": start, "label": label, "conf": conf,
                "fps": fps, "dur": dur
            }

    cap.release()

    if best["conf"] < 0.0:
        return None, "❌ 유효한 창을 찾지 못했습니다.", {"scores": all_scores, "dur": dur}

    cap = cv2.VideoCapture(video_path)
    thumb = read_frame_at_time(cap, best["fps"], best["start"])
    cap.release()
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB) if thumb is not None else None

    msg = f"✅ best start={best['start']:.2f}s | label={best['label']} | conf={best['conf']:.2f} | dur={best['dur']:.2f}s"
    return thumb_rgb, msg, {"scores": all_scores, "dur": best["dur"]}

# =========================
# (B) 실시간: 4-phase 중첩 윈도우 (0.25s step) + Threshold + Cooldown
# =========================
DT_STEP = 0.25       # 0.25초마다 1장 채택 (4Hz)
PHASES = 4           # 4개 위상 => 시작 오프셋 0.25s씩
WIN = 16             # 각 위상 버퍼 길이(프레임)
T_ON  = 0.60         # 트리거 임계
T_OFF = 0.50         # 재무장 임계(히스테리시스)
COOLDOWN_SEC = 3.0   # 강제 휴식
STABLE_K = 1         # 연속 K회 이상 T_ON 넘으면 트리거

monotonic = time.monotonic

rt_state = {
    "bufs": [deque(maxlen=WIN) for _ in range(PHASES)],  # 각 위상 버퍼
    "last_tick": None,        # 마지막 샘플 시각
    "phase": 0,               # 현재 위상(0..3)
    "cooldown_until": 0.0,
    "stable_cnt": 0,
    "armed": True,
    "last_pred": None,        # (label, prob, phase)
}

def draw_overlay(img_bgr, text, color=(0,255,0), y=30, scale=0.9, thick=2):
    out = img_bgr.copy()
    cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return out

def realtime_fn(frame_rgb):
    # Gradio webcam 입력은 RGB
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    now = monotonic()

    # 0.25초마다 1장만 채택
    if rt_state["last_tick"] is None or (now - rt_state["last_tick"]) >= DT_STEP - 1e-3:
        # 현재 위상 결정(순환)
        ph = rt_state["phase"]
        # 포즈 추출
        kps = frame_to_keypoints_bgr(frame_bgr)  # (2,V)
        # 해당 위상 버퍼에만 push (=> 각 버퍼는 1초 간격으로 채워짐)
        rt_state["bufs"][ph].append(kps)
        rt_state["last_tick"] = now
        rt_state["phase"] = (ph + 1) % PHASES

        # 쿨다운 중이면 추론 생략하고 오버레이만
        if now < rt_state["cooldown_until"]:
            remain = rt_state["cooldown_until"] - now
            overlay = draw_overlay(frame_bgr, f"Cooldown: {remain:.1f}s | phase={ph} len={len(rt_state['bufs'][ph])}", (128,200,255))
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # 해당 위상 버퍼가 16프레임을 채웠다면 이 위상 윈도우로 추론 실행
        if len(rt_state["bufs"][ph]) == WIN:
            seq = list(rt_state["bufs"][ph])  # (2,V) * 16
            joints, bones = pack_for_agcn(seq)
            ex_pred, ex_prob, _ = agcn_forward(joints, bones)
            label = int(ex_pred[0].item())
            prob  = float(ex_prob[0, label].item())
            rt_state["last_pred"] = (label, prob, ph)

            # 히스테리시스 + 안정 카운트
            thresh = T_ON if rt_state["armed"] else T_OFF
            if prob >= thresh:
                rt_state["stable_cnt"] += 1
            else:
                rt_state["stable_cnt"] = 0

            if rt_state["stable_cnt"] >= STABLE_K:
                # 트리거
                overlay_bgr = draw_overlay(frame_bgr, f"DETECTED! label={label} p={prob:.2f} (phase={ph})", (0,0,255), y=60, scale=1.0, thick=3)
                rt_state["cooldown_until"] = now + COOLDOWN_SEC
                rt_state["stable_cnt"] = 0
                rt_state["armed"] = False
                return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            else:
                if prob < T_OFF:
                    rt_state["armed"] = True

    # 기본 오버레이: 최근 추론 결과(있으면) 표시
    txt = "Warming "  # 어떤 위상이든 16장 미만이면 warming일 수 있음
    if rt_state["last_pred"] is not None:
        l, p, ph = rt_state["last_pred"]
        txt = f"pred={l} p={p:.2f} (phase={ph})"
    overlay = draw_overlay(frame_bgr, txt, (0,255,0))
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# =========================
# Gradio UI
# =========================
# gradio v4 기준: theme 객체가 더 안전하지만 문자열도 허용될 수 있음
with gr.Blocks(title="Video (Offset Scan) & Realtime (Overlapped 0.25s)") as demo:
    gr.Markdown("## 업로드 영상(오프셋 스캔) & 실시간(4-Phase Overlapped 0.25s) 데모")
    with gr.Tab("업로드 영상 / 오프셋 스캔"):
        gr.Markdown(
            "- 영상은 **길이 16초 이상**이어야 합니다. (1Hz×16 프레임)\n"
            "- 시작 오프셋을 **0.25초 간격**으로 이동하며 각 창(16초)을 평가하고, "
            "**가장 확신(conf)** 이 높은 구간을 선택합니다."
        )
        in_video = gr.Video(sources='upload', label="영상 입력")
        offset_step = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="오프셋 간격(초)")
        conf_floor  = gr.Slider(0.0, 0.99, value=0.0, step=0.01, label="최소 확신(conf) 하한 (선택)")
        run_btn = gr.Button("스캔 실행")
        thumb = gr.Image(label="대표 썸네일(최고 구간의 첫 프레임)")
        msg   = gr.Textbox(label="결과 메시지", lines=2)
        dbg   = gr.JSON(label="디버그: 창별 확신 리스트/길이(sec)")

        def run_scan(video, step, floor):
            img, text, meta = scan_video_offsets(video, offset_step=step, conf_floor=floor)
            return img, text, meta

        run_btn.click(run_scan, inputs=[in_video, offset_step, conf_floor], outputs=[thumb, msg, dbg])

    with gr.Tab("실시간 / 4-Phase Overlap (0.25s)"):
        gr.Markdown(
            "- **0.25초마다** 프레임 1장을 채택하고, 4개 위상(phase=0..3) 버퍼에 **라운드로빈**으로 분배합니다.\n"
            "- 각 버퍼는 **길이 16**을 채우면 즉시 추론됩니다. (결과적으로 시작점이 0.25초씩 민감해짐)\n"
            "- 히스테리시스(T_on/T_off)와 쿨다운 시간은 아래에서 조절할 수 있습니다."
        )
        cam = gr.Image(sources="webcam", streaming=True, label="웹캠 입력")
        with gr.Row():
            T_on  = gr.Slider(0.3, 0.95, value=T_ON,  step=0.01, label="T_on (트리거)")
            T_off = gr.Slider(0.2, 0.90, value=T_OFF, step=0.01, label="T_off (재무장)")
            cd    = gr.Slider(1, 5, value=COOLDOWN_SEC, step=1, label="쿨다운(초)")
        out_img = gr.Image(label="실시간 출력")

        # 파라미터 라이브 반영용 래퍼
        def realtime_wrapper(img, t_on, t_off, cooldown):
            global T_ON, T_OFF, COOLDOWN_SEC
            T_ON = float(t_on); T_OFF = float(t_off); COOLDOWN_SEC = float(cooldown)
            return realtime_fn(img)

        cam.stream(realtime_wrapper, inputs=[cam, T_on, T_off, cd], outputs=out_img,
                   stream_every=0.1, concurrency_limit=1, time_limit=600)

    gr.Markdown("> 참고: 포즈 추출 실패 시 현재는 **랜덤 키포인트 fallback**(데모 목적)으로 동작합니다.")

if __name__ == "__main__":
    demo.launch()
