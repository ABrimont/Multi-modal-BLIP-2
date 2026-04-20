from huggingface_hub import hf_hub_download
import torch

repo_id = "DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned"
ckpt_path = hf_hub_download(repo_id=repo_id, filename="AL_LLaMA_2_7B_Finetuned.pth")

state_dict = torch.load(ckpt_path, map_location="cpu")
print("Nombre total de clés:", len(state_dict))

print(state_dict.keys())

ca_keys = [k for k in state_dict["model"].keys() if "cross" in k.lower() or "EncDecAttention" in k]
print("Couches Cross-Attention trouvées :", len(ca_keys))
for k in ca_keys:
    print(k)

# Vérifier si 'query_tokens' est présent
if "query_tokens" in state_dict["model"]:
    q_tokens = state_dict["model"]["query_tokens"]
    print("Query tokens trouvés :", q_tokens.shape)
else:
    # Chercher toutes les clés qui contiennent "query"
    q_keys = [k for k in state_dict["model"].keys() if "query" in k.lower()]
    print("Clés candidates pour query tokens :", q_keys)
    if q_keys:
        q_tokens = state_dict["model"][q_keys[0]]
        print("Query tokens shape:", q_tokens.shape)
