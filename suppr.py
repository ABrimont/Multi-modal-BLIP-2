import os
import json

bad_videos = {
    "c3DAquBQ2dg",
    "uH4TF9gl21I",
    "7olz18uNTUI",
    "p5gYu1up1Ac",
    "6L5qrX6aKCA",
}

ann_dir = "/home/abrimont/partage/mllm-video-captioner/lavis/datasets/vatex/annotations"

for fname in os.listdir(ann_dir):
    if fname.endswith(".json"):
        path = os.path.join(ann_dir, fname)

        with open(path, "r") as f:
            data = json.load(f)

        cleaned = [ann for ann in data if ann.get("video") not in bad_videos]

        print(f"{fname}: {len(data)} → {len(cleaned)} après nettoyage")

        with open(path, "w") as f:
            json.dump(cleaned, f, indent=2)
