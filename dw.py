"""
Download pretrained weights for Multi-modal-BLIP-2
Weights are hosted on Hugging Face Hub (~10.2GB each)
"""

from huggingface_hub import hf_hub_download
import os

# Directory where weights will be saved
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODELS = [
    {
        "repo_id": "AntBri/VATEX_weights",
        "filename": "VATEX_best.pth",
        "description": "VATEX dataset weights",
    },
    {
        "repo_id": "AntBri/MSRVTT_weights",
        "filename": "MSRVTT_best.pth",
        "description": "MSRVTT dataset weights",
    },
    {
        "repo_id": "AntBri/AudioCaps_weights",
        "filename": "AudioCaps_best.pth",
        "description": "AudioCaps dataset weights",
    },
]


def download_weights(models=MODELS):
    print("Downloading Multi-modal-BLIP-2 weights from Hugging Face...\n")

    for model in models:
        dest = os.path.join(WEIGHTS_DIR, model["filename"])

        if os.path.exists(dest):
            print(f"✓ {model['filename']} already exists, skipping.")
            continue

        print(f"⬇ Downloading {model['description']} ({model['filename']})...")
        hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=WEIGHTS_DIR,
        )
        print(f"✓ Saved to {dest}\n")

    print("All weights downloaded successfully!")
    print(f"Location: {os.path.abspath(WEIGHTS_DIR)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Multi-modal-BLIP-2 weights")
    parser.add_argument(
        "--model",
        choices=["VATEX", "MSRVTT", "AudioCaps", "all"],
        default="all",
        help="Which weights to download (default: all)",
    )
    args = parser.parse_args()

    if args.model == "all":
        download_weights()
    else:
        selected = [m for m in MODELS if m["filename"].startswith(args.model)]
        download_weights(selected)
