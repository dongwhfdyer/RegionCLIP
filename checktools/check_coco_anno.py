import json
from pathlib import Path

f0 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_all.json")
f1 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_b.json")
f2 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_t.json")
f3 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_val2017_all.json")
f4 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_val2017_b.json")
f5 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_val2017_t.json")
f6 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/10images.json")


data1 = json.load(open(f0, "r"))
data2 = json.load(open(f1, "r"))
data3 = json.load(open(f2, "r"))
data4 = json.load(open(f3, "r"))
data5 = json.load(open(f4, "r"))
data6 = json.load(open(f5, "r"))
data7 = json.load(open(f6, "r"))



print("--------------------------------------------------")

