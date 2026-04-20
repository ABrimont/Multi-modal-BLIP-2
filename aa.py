import torch
import itertools
import numpy as np
from pycocoevalcap.cider.cider import Cider


# ============ Données artificielles ============
B = 1           # batch size = 1 vidéo
K = 3           # beams
L = 10          # longueur tokens (fake)

# 3 "séquences de tokens" (beams)
topk_ids = torch.tensor([
    [ [101, 2009, 2003, 1037, 2711, 102, 0, 0, 0, 0],
      [101, 2009, 2003, 1037, 2171, 102, 0, 0, 0, 0],
      [101, 2009, 2003, 1037, 2265, 102, 0, 0, 0, 0] ]
])  # shape: (B, K, L)

# fake logprobs par token
topk_logprobs = torch.randn(B, K, L)

# Ground truth unique pour cette vidéo
gt_caps = ["a man is riding a bike"]

# ============ 1. Décodage textes beams ============
# Ici pas de tokenizer → texte artificiel
gen_texts = [
    "it is a bicycle",
    "it is a person",
    "it is a vehicle",
]

print("🎯 GEN TEXTS :", gen_texts)
print("🎯 GT :", gt_caps)


# ============ 2. Structure caps_gen (répétée pour chaque GT) ============
caps_gen = list(itertools.chain.from_iterable(
    [[c] * len(gt_caps) for c in gen_texts]
))
print("\n✅ caps_gen =", caps_gen)


# ============ 3. Structure caps_gt (répétée pour chaque beam) ============
caps_gt = []
for k in range(K):
    for gt in gt_caps:
        caps_gt.append(gt)

print("✅ caps_gt  =", caps_gt)


# ============ 4. Conversion en format CIDEr ============
def tokenize_cider(refs, hyps):
    refs_dict = {}
    hyps_dict = {}
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        refs_dict[i] = [ref]   # liste !
        hyps_dict[i] = [hyp]   # liste !
    return refs_dict, hyps_dict

refs, hyps = tokenize_cider(caps_gt, caps_gen)

print("\n📦 refs =", refs)
print("📦 hyps =", hyps)


# ============ 5. Calcul CIDEr ============
cider = Cider()
mean_score, scores = cider.compute_score(refs, hyps)

scores = np.array(scores)       # shape (K)
scores_t = torch.tensor(scores).view(B, K)

print("\n🥇 CIDEr par beam =", scores)
print("📏 Shape =", scores_t.shape)


# ============ 6. Loss SCST ============
reward = scores_t                      # shape (1, 3)
baseline = reward.mean(dim=1, keepdim=True)
advantage = reward - baseline          # shape (1, 3)
logprobs_mean = topk_logprobs.mean(dim=2)   # shape (1, 3)

loss = - (advantage.detach() * logprobs_mean).mean()

print("\n✅ Baseline =", baseline)
print("✅ Advantage =", advantage)
print("✅ Logprobs mean =", logprobs_mean)
print("\n🔥 SCST Loss =", loss.item())
