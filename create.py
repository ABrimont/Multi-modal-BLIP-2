import json
import os
import cv2
import torchaudio
from random import shuffle

# --- Config ---
intern_jsonl = '/home/abrimont/partage/InternVideo2/InternVideo2-YTT-AVS.jsonl'
intern_dir = '/home/abrimont/partage/InternVideo2/raw_videos/'
output_path_raw = "/home/abrimont/partage/mllm-video-captioner/PT.json"
output_path_clean = "/home/abrimont/partage/mllm-video-captioner/PT_clean.json"

# --- Fonction de vérification vidéo/audio ---
def is_video_readable(video_path):
    # Test ouverture vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return False
    
    # Test ouverture audio
    try:
        torchaudio.load(video_path)
    except Exception:
        return False

    return True

# --- Chargement des métadonnées ---
intern_data = []
with open(intern_jsonl, 'r') as f:
    for line in f:
        intern_data.append(json.loads(line))

# --- Construction de la liste brute ---
output = []
current_id = 0

intern_files = os.listdir(intern_dir)
for fname in intern_files:
    base = fname.rsplit('.', 1)[0]
    parts = base.split('_')
    id_debut = '_'.join(parts[:-1])
    dernier_nombre = parts[-1]

    try:
        video_info = next(
            item for item in intern_data
            if item.get("YoutubeID") == id_debut and item.get("start_frame") == int(dernier_nombre)
        )
    except StopIteration:
        continue

    caption = video_info['avs']  # 'avs', 'visual', 'audio', etc.
    print(video_info)
    output.append({
        "image_id": fname,
        "caption": caption,
        "id": current_id,
        "video": os.path.join(intern_dir, fname)
    })
    current_id += 1

# --- Mélange ---
shuffle(output)

# --- Sauvegarde brute ---
with open(output_path_raw, 'w') as f:
    json.dump(output, f, indent=2)
print(f"[INFO] JSON brut sauvegardé avec {len(output)} entrées : {output_path_raw}")

# --- Nettoyage : suppression vidéos absentes, vides ou corrompues ---
cleaned_data = []
removed = []

for item in output:
    video_path = item.get("video")
    if (not video_path 
        or not os.path.exists(video_path) 
        or os.path.getsize(video_path) == 0 
        or not is_video_readable(video_path)):
        removed.append(video_path)
    else:
        cleaned_data.append(item)

print(f"[INFO] {len(removed)} vidéos supprimées car absentes, vides ou corrompues.")
for v in removed:
    print(f" - {v}")

# --- Sauvegarde finale ---
with open(output_path_clean, "w") as f:
    json.dump(cleaned_data, f, indent=2)
print(f"[INFO] JSON nettoyé sauvegardé avec {len(cleaned_data)} entrées : {output_path_clean}")
