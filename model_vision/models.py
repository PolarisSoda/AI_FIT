from ultralytics import YOLO
import torch,os

if __name__ == '__main__':
    DATA = "data.yaml"
    model = YOLO("yolo11n-pose.pt")
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    model = YOLO("yolo11s-pose.pt")
    results1 = model.train(
        data=DATA,
        imgsz=960, 
        epochs=15,
        batch=16,
        device=DEVICE,
        lr0=5e-4, lrf=0.1,
        warmup_epochs=3,
        amp=True,
        freeze="backbone",
        workers=8,
    )

    ckpt = model.ckpt_path or "runs/pose/train/weights/last.pt"

    model2 = YOLO(ckpt)
    results2 = model2.train(
        data=DATA,
        imgsz=960,
        epochs=40,
        batch=16,
        device=DEVICE,
        lr0=3e-4, lrf=0.05,
        warmup_epochs=1,
        amp=True,
        workers=8,
    )

    val_res = model2.val(data=DATA, imgsz=960, device=DEVICE)
    print(val_res)