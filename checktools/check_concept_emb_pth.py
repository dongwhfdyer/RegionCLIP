from pathlib import Path
import torch


f1 = Path("/data/pcl/RegionCLIP/pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth")
f2 = Path("/data/pcl/RegionCLIP/pretrained_ckpt/concept_emb/coco_nouns_4764_emb.pth")
f3 = Path("/data/pcl/RegionCLIP/pretrained_ckpt/concept_emb/dior_all_classes_embeds.pth")
d1 = torch.load(f1)
d2 = torch.load(f2)
d3 = torch.load(f3)
print("--------------------------------------------------")

