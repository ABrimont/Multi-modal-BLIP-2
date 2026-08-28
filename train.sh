# AudioCaps
torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train/AudioCaps_stage1.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train/AudioCaps_stage2.yaml




# msrvtt
torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train/msrvtt_stage1.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train/msrvtt_stage2.yaml




# VATEX 
torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train/VATEX_stage1.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train/VATEX_stage2.yaml


# Variants msrvtt
torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_Aligned_6.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_Aligned_12.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_Alternating.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_Increased.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_Random_Init.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_Separate.yaml

torchrun --standalone --nproc_per_node=2 \
  train.py --cfg-path lavis/projects/blip2/train_variants_msrvtt/msrvtt_unimodal.yaml

