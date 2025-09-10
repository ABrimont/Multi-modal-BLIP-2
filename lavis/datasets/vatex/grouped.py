import json
import os
from collections import defaultdict

# chemins
in_file = "/home/abrimont/partage/mllm-video-captioner/lavis/datasets/vatex/annotations/train_clean.json"
out_file = "/home/abrimont/partage/mllm-video-captioner/lavis/datasets/vatex/annotations/train_grouped.json"

# charger données
with open(in_file, "r") as f:
    data = json.load(f)

# regroupement par image_id
grouped = defaultdict(lambda: {"caption": []})

for ann in data:
    vid = ann["image_id"]
    if "image_id" not in grouped[vid]:
        grouped[vid]["image_id"] = vid
        grouped[vid]["video"] = ann.get("video", vid)
        grouped[vid]["id"] = len(grouped) - 1  # index unique basé sur ordre d'apparition
    grouped[vid]["caption"].append(ann["caption"])

# conversion en liste
out_data = list(grouped.values())

# sauvegarde
with open(out_file, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"✅ Sauvé {len(out_data)} vidéos regroupées dans {out_file}")
