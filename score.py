import json
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def tokenize(refs, cands):
    tokenizer = PTBTokenizer()
    refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
    cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}
    refs = tokenizer.tokenize(refs)
    cands = tokenizer.tokenize(cands)
    return refs, cands


def pycoco_eval(scorer, refs, cands):
    refs, cands = tokenize(refs, cands)
    avg_score, scores = scorer.compute_score(refs, cands)
    return avg_score, scores


def get_all_metrics(refs, cands):
    scorers = [
        (Bleu(4), "bleu"),
        (Meteor(), "meteor"),
        (Rouge(), "rouge"),
        (Cider(), "cider"),
    ]
    metrics = {}
    for scorer, name in scorers:
        overall, _ = pycoco_eval(scorer, refs, cands)
        metrics[name] = overall
    return metrics


def caption_eval(annotation_file, results_file, use_video_id=True):
    with open(results_file, "r") as f:
        results = json.load(f)

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    id_key = "video_id" if use_video_id else "image_id"

    ids = []
    candidates = {}
    for res in results:
        candidates[res[id_key]] = res["caption"]
        ids.append(res[id_key])

    references = {}
    for ann in annotations["annotations"]:
        key = ann.get("video_id", ann.get("image_id"))
        if key not in references:
            references[key] = []
        references[key].append(ann["caption"])

    # aligner sur les ids présents dans les prédictions
    candidates = [candidates[cid] for cid in ids]
    references = [references[cid] for cid in ids]

    metrics = get_all_metrics(references, candidates)

    print(f"===== Scores ({len(ids)} items évalués) =====")
    for k, v in metrics.items():
        if k == "bleu":
            for bidx, sc in enumerate(v):
                print(f"BLEU-{bidx+1}: {sc:.4f}")
        else:
            print(f"{k.upper()}: {v:.4f}")

    return metrics


# if __name__ == "__main__":
#     anno_file = "/home/abrimont/partage/VALOR/datasets/vatex/caption_anno_en.json"
#     pred_file = "/home/abrimont/partage/VAST/output/vast/pretrain_vast/downstream/caption-vatex/results_test_vatex_cap/step_33939_tvas.json"     

#     caption_eval(anno_file, pred_file, use_video_id=True)

if __name__ == "__main__":
    anno_file = "/home/abrimont/partage/VALOR/datasets/vatex/caption_anno_en.json"
    pred_file = "/home/abrimont/partage/mllm-video-captioner/lavis/output/Video-BLIP2/FLAN-T5-XL/Caption_vatex_audio/20250916012/result/val_epoch2.json"
    caption_eval(anno_file, pred_file,  use_video_id=False)
