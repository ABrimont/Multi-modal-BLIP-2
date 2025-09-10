import os
import json

# chemins
ann_file = "/home/abrimont/partage/mllm-video-captioner/lavis/datasets/vatex/annotations/cap_train.json"
video_root = "/home/abrimont/partage/VALOR/datasets/vatex/raw_videos"
out_file = ann_file.replace(".json", "_superclean.json")

# charger annotations
with open(ann_file, "r") as f:
    anns = json.load(f)

# lister vidéos valides présentes sur disque
valid_videos = set(os.listdir(video_root))  # ex: "77ntFzO_sBs.mp4"

cleaned = []
for ann in anns:
    # extraire l'ID de base : "xxxxxx.mp4"
    base = ann["video"].split("/")[-1]      # ex "77ntFzO_sBs_000057_000067.mp4"
    vid_id = base.split("_")[0] + ".mp4"    # ex "77ntFzO_sBs.mp4"

    # garder seulement si la vidéo existe dans raw_videos
    if vid_id in valid_videos:
        cleaned.append({
            "image_id": vid_id,
            "video": vid_id,
            "caption": ann.get("caption", ""),
            "id": len(cleaned)  # réindex propre
        })

print(f"Avant: {len(anns)}  → Après nettoyage: {len(cleaned)}")

with open(out_file, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"Fichier sauvegardé: {out_file}")
