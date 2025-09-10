import json
import re

# chemin vers ton fichier original
input_file = "/home/abrimont/partage/mllm-video-captioner/lavis/datasets/vatex/annotations/cap_test_filtered.json"
output_file = input_file.replace(".json", "_clean.json")

with open(input_file, "r") as f:
    data = json.load(f)

for entry in data:
    if "video" in entry:
        old_path = entry["video"]

        # garder juste l'ID avant le premier "_000" et ajouter .mp4
        video_id = re.sub(r"_\d{6}_\d{6}\.mp4$", "", old_path.split("/")[-1])
        new_path = f"{video_id}.mp4"

        entry["video"] = new_path
        entry["image_id"] = new_path  # si tu veux aussi modifier image_id

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Fichier sauvegardé dans {output_file}")
