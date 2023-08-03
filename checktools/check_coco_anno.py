import json
from pathlib import Path

f0 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_all.json")
f1 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_b.json")
f2 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_t.json")
f3 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_val2017_all.json")
f4 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_val2017_b.json")
f5 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_val2017_t.json")

with open(f0, "r") as f:
    data1 = json.load(f)
    print("--------------------------------------------------")
