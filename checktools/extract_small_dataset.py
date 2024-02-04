# #---------kkuhn-block------------------------------ # version 1
# import json
# import random
# from pathlib import Path
#
# f1 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_b.json")
#
# with open(f1, "r") as f:
#     data = json.load(f)
#
# num_images = 10
# selected_images = random.sample(data["images"], num_images)
#
# selected_data = data.copy()
# selected_data["images"] = selected_images
# selected_image_ids = [img["id"] for img in selected_images]
#
# # iterate over annotations and remove those that are not in selected_images
# selected_anns = []
# for ann in data["annotations"]:
#     if ann["image_id"] in selected_image_ids:
#         selected_anns.append(ann)
# selected_data["annotations"] = selected_anns
#
#
# with open("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/10images.json", "w") as f:
#     json.dump(selected_data, f)
#
#
# with open("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/10images.json", "r") as f:
#     data = json.load(f)
# print("--------------------------------------------------")
#
# #---------kkuhn-block------------------------------


import json
import random
from pathlib import Path

f1 = Path("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/ovd_ins_train2017_b.json")

with open(f1, "r") as f:
    data = json.load(f)

num_images = 10
selected_images = random.sample(data["images"], num_images)

selected_data = data.copy()
selected_data["images"] = selected_images
selected_image_ids = [img["id"] for img in selected_images]

# iterate over annotations and remove those that are not in selected_images
selected_anns = []
for ann in data["annotations"]:
    if ann["image_id"] in selected_image_ids:
        selected_anns.append(ann)
selected_data["annotations"] = selected_anns

# add missing keys to selected_data
selected_data.setdefault("licenses", [])
selected_data.setdefault("info", {})
selected_data.setdefault("type", "instances")

with open("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/10images.json", "w") as f:
    json.dump(selected_data, f)

with open("/data/pcl/proj/RegionCLIP/datasets/coco/annotations/10images.json", "r") as f:
    data = json.load(f)
print("--------------------------------------------------")
