"""
sign_types.py
-------------
Which traffic signs the car passes — the junction signs above all (yield, stop,
priority road, right of way at the next junction, roundabout), a cue that a junction
is ahead and of which kind.

    where    Mask2Former (Mapillary Vistas) "traffic sign (front)" regions, every 6th frame
             (5 per second, the frames perception analyses), from the model-input video
             dense.py uploads
    which    a ResNet-18 trained on GTSRB (German Traffic Sign Recognition Benchmark,
             43 classes; Moldova's signs follow the same Vienna Convention), on Colab, once —
             the weights are kept in models/ and reused for every drive

Crops of the recognised signs are saved (sign_crops.zip) to check by eye.

Usage:
    .venv-video/bin/python -m src.video.sign_types <drive>        # Colab T4

Output: data/video/processed/<drive>/sign_types.parquet
        (t, x, y, w, h in the 960-px image, cls, name, prob, junction)
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.video import drives
from src.video.dense import FRAME_STEP, input_height, input_video
from src.video.drives import ROOT
from src.video.scene import CLASSES

WEIGHTS = ROOT / "models" / "gtsrb_resnet18.pt"
GTSRB = ["speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60", "speed limit 70", "speed limit 80",
         "end of speed limit 80", "speed limit 100", "speed limit 120", "no overtaking", "no overtaking (trucks)",
         "right of way at next junction", "priority road", "yield", "stop", "no vehicles", "no trucks", "no entry",
         "general caution", "dangerous curve left", "dangerous curve right", "double curve", "bumpy road",
         "slippery road", "road narrows right", "road works", "traffic signals", "pedestrians", "children crossing",
         "bicycles crossing", "ice / snow", "wild animals", "end of all limits", "turn right ahead", "turn left ahead",
         "ahead only", "ahead or right", "ahead or left", "keep right", "keep left", "roundabout",
         "end of no overtaking", "end of no overtaking (trucks)"]
JUNCTION = {11, 12, 13, 14, 17, 26, 33, 34, 35, 36, 37, 38, 39, 40}   # announce or regulate a junction
MIN_PROB = 0.85

REMOTE = '''
import io, json, os, time, zipfile
import cv2, numpy as np, pandas as pd, torch, torchvision
from torch import nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
dev = "cuda"
SIZE = 64
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def classifier():
    m = torchvision.models.resnet18()
    m.fc = nn.Linear(512, 43)
    return m

model = classifier().to(dev)
if os.path.exists("/content/gtsrb.pt"):
    model.load_state_dict(torch.load("/content/gtsrb.pt", map_location=dev))
    print("classifier: reusing trained weights", flush=True)
else:                                               # train once on GTSRB
    aug = T.Compose([T.Resize((SIZE, SIZE)), T.RandomResizedCrop(SIZE, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                     T.ColorJitter(0.5, 0.5, 0.4, 0.05), T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.4),
                     T.ToTensor(), norm])
    plain = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), norm])
    train = torchvision.datasets.GTSRB("/content/gtsrb", split="train", download=True, transform=aug)
    test = torchvision.datasets.GTSRB("/content/gtsrb", split="test", download=True, transform=plain)
    print("GTSRB:", len(train), "training,", len(test), "test images", flush=True)
    m0 = torchvision.models.resnet18(weights="IMAGENET1K_V1"); m0.fc = nn.Linear(512, 43); model = m0.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=2)
    EPOCHS = 5
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * len(dl))
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda"):
                loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"epoch {{ep + 1}}/{{EPOCHS}} loss {{tot / len(dl):.3f}}", flush=True)
    model.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(test, batch_size=512, num_workers=2):
            ok += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    print(f"GTSRB test accuracy {{ok / n:.4f}}", flush=True)
    torch.save(model.state_dict(), "/content/gtsrb.pt")
model.eval()

CLASSES = {classes}
seg_proc = AutoImageProcessor.from_pretrained("{seg}")
seg = Mask2FormerForUniversalSegmentation.from_pretrained("{seg}").eval().cuda().half()
sign_ids = [i for i, n in seg.config.id2label.items() if CLASSES.get(n) == "sign_front"]
prep = T.Compose([T.ToPILImage(), T.Resize((SIZE, SIZE)), T.ToTensor(), norm])
cap = cv2.VideoCapture("/content/input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS); n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); n = (n_all + {step} - 1) // {step}
rows, crops, f, k, t0 = [], zipfile.ZipFile("/content/sign_crops.zip", "w"), 0, 0, time.time()
while cap.grab():
    if f % {step} == 0:
        img = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2RGB)[:{img_h}]
        t = f / fps
        inp = {{kk: v.cuda().half() if v.dtype.is_floating_point else v.cuda()
               for kk, v in seg_proc(images=img, return_tensors="pt").items()}}
        with torch.no_grad():
            out = seg(**inp)
        out.class_queries_logits = out.class_queries_logits.float(); out.masks_queries_logits = out.masks_queries_logits.float()
        s = seg_proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].cpu().numpy()
        mask = np.isin(s, sign_ids).astype(np.uint8)
        nb, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = [stats[i] for i in range(1, nb) if stats[i][4] >= 80 and min(stats[i][2], stats[i][3]) >= 9
                 and 0.4 <= stats[i][2] / stats[i][3] <= 2.5]
        if boxes:
            batch = []
            for x, y, w, h, a in boxes:
                m = int(0.15 * max(w, h))
                c = img[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m]
                batch.append(prep(c))
            with torch.no_grad():
                p = torch.softmax(model(torch.stack(batch).to(dev)), 1).cpu().numpy()
            for (x, y, w, h, a), pr in zip(boxes, p):
                c = int(pr.argmax())
                rows.append(dict(t=round(t, 2), x=int(x), y=int(y), w=int(w), h=int(h), cls=c, prob=float(pr[c])))
                if pr[c] >= 0.6 and len(crops.namelist()) < 3000:
                    m = int(0.15 * max(w, h))
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m], cv2.COLOR_RGB2BGR))
                    crops.writestr(f"{{t:07.2f}}_{{c:02d}}_{{int(pr[c] * 100)}}.jpg", buf.tobytes())
        k += 1
        if k % 300 == 0 or k == n:
            el = time.time() - t0
            print(f"signs {{k}}/{{n}} frames, {{len(rows)}} signs  {{el / k:.2f}} s/frame  ~{{el / k * (n - k) / 60:.1f}} min left", flush=True)
    f += 1
crops.close()
pd.DataFrame(rows).to_parquet("/content/sign_types.parquet", index=False)
print("done", len(rows), "signs", flush=True)
'''


def run_colab(drv: dict, reuse: bool = False):
    from src.video.colab import Session
    folder, key = drv["folder"], drv["key"]
    video = input_video(drv)
    script = REMOTE.format(classes=json.dumps(CLASSES), seg="facebook/mask2former-swin-large-mapillary-vistas-semantic",
                           step=FRAME_STEP, img_h=input_height(drv))
    t0 = time.time()
    with Session(f"signs-{key}", reuse=reuse) as s:
        if WEIGHTS.exists():
            s.upload(WEIGHTS, "/content/gtsrb.pt")
        s.upload(video, "/content/input.mp4")
        s.install("transformers==4.46.3")
        s.start(script)
        if s.wait() != 0:
            print(s.log())
            raise RuntimeError("sign recognition on Colab failed")
        if not WEIGHTS.exists():
            s.download("/content/gtsrb.pt", WEIGHTS)
        s.download("/content/sign_types.parquet", folder / "sign_types.parquet")
        s.download("/content/sign_crops.zip", folder / "sign_crops.zip")
    df = label(pd.read_parquet(folder / "sign_types.parquet"))
    df.to_parquet(folder / "sign_types.parquet", index=False)
    sure = df[df.prob >= MIN_PROB]
    print(f"→ {folder / 'sign_types.parquet'}: {len(df)} sign regions, {len(sure)} recognised (≥ {MIN_PROB:.0%}), "
          f"{sure.junction.sum()} junction signs; Colab {(time.time() - t0) / 60:.1f} min", flush=True)
    print(sure.groupby("name").size().sort_values(ascending=False).head(15).to_string(), flush=True)


def label(df: pd.DataFrame) -> pd.DataFrame:
    df["name"] = [GTSRB[c] for c in df.cls]
    df["junction"] = df.cls.isin(JUNCTION)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--reuse", action="store_true", help="attach to the drive's runtime if still alive")
    a = ap.parse_args()
    run_colab(drives.load(a.drive), a.reuse)


if __name__ == "__main__":
    main()
