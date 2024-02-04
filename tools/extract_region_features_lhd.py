#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import os
import torch
from torch.nn import functional as F
import numpy as np
import time
import json
import pickle
from detectron2.structures import Boxes

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.clip_rcnn import visualize_proposals
import cv2
from tqdm import tqdm


def visualize_bbox(image_name, bbox_list, lable_image_path):
    file_name = image_name.split("/")[-1]
    image = cv2.imread(image_name)
    # resized_image = cv2.resize(image, None, fx=scale_factor[0]/256, fy=scale_factor[1]/256)
    for bbox in bbox_list[: 1]:
        x1, y1, x2, y2 = bbox  # 假设bbox是一个四元组 [x1, y1, x2, y2]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绘制矩形框

    # cv2.imshow("Image with Bboxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(lable_image_path+"image_labled", file_name), image)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_inputs(cfg, file_name):
    """ Given a file name, return a list of dictionary with each dict corresponding to an image
    (refer to detectron2/data/dataset_mapper.py)
    """
    # image loading
    dataset_dict = {}
    image = utils.read_image(file_name, format=cfg.INPUT.FORMAT)
    dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]  # h, w before transforms

    # image transformation
    augs = utils.build_augmentation(cfg, False)
    augmentations = T.AugmentationList(
        augs)  # [ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')]
    aug_input = T.AugInput(image)
    transforms = augmentations(aug_input)
    image = aug_input.image
    h, w = image.shape[:2]  # h, w after transforms
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    return [dataset_dict]


def create_model(cfg):
    """ Given a config file, create a detector
    (refer to tools/train_net.py)
    """
    # create model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
            and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None \
            and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN':  # load 2nd pretrained model
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )

    assert model.clip_crop_region_type == "RPN"
    assert model.use_clip_c4  # use C4 + resnet weights from CLIP
    assert model.use_clip_attpool  # use att_pool from CLIP to match dimension
    model.roi_heads.box_predictor.vis = True  # get confidence scores before multiplying RPN scores, if any
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model


def scale_bbox(bbox_list, scale_factor):
    scaled_bbox_list = []
    for bbox in bbox_list:
        scaled_bbox = [coord * scale_factor for coord in bbox]
        scaled_bbox_list.append(scaled_bbox)
    return scaled_bbox_list


def extract_region_feats(cfg, model, batched_inputs, file_name):
    """ Given a model and the input images, extract region features and save detection outputs into a local file
    (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
    """
    # model inference
    # 1. localization branch: offline modules to get the region proposals
    images = model.offline_preprocess_image(batched_inputs)
    features = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, features, None)
    # visualize_proposals(batched_inputs, proposals, model.input_format)

    # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)
    # with open("/data/pcl/RegionCLIP/box_pkl/15_storage tank,stock tank,tank_9.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     # a = 800 / 256
    # data = scale_bbox(data["boxes"], 800 / 256)

    # visualize_bbox("/data/pcl/RegionCLIP/datasets/custom_images/15_storage tank,stock tank,tank_9.jpg", data)


    # detectron2_boxes = [Boxes(data + [data[-1]] * (1000 - len(data))).to('cuda')]
    # 3. given the proposals, crop region features from 2D image features
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features], proposal_boxes, model.backbone.layer4
    )
    att_feats = model.backbone.attnpool(box_features)  # region features
    # ___________________________lihd_block________________________________________________
    #     pre_computed_w = torch.load("/data/pcl/RegionCLIP/output/concept_feats/concept_embeds2.pth")
    #     normalized_x = F.normalize(att_feats, p=2.0, dim=1)
    #     cls_scores = normalized_x.cpu() @ F.normalize(pre_computed_w, p=2.0, dim=1).t()
    #     pred_probs = F.softmax(cls_scores, dim=-1)[
    #         keep_indices[im_id]]

    # ___________________________lihd_block________________________________________________

    if cfg.MODEL.CLIP.TEXT_EMB_PATH is None:  # save features of RPN regions
        results = model._postprocess(proposals, batched_inputs)  # re-scale boxes back to original image size

        # save RPN outputs into files
        im_id = 0  # single image
        pred_boxes = results[im_id]['instances'].get("proposal_boxes").tensor  # RPN boxes, [#boxes, 4]
        region_feats = att_feats  # region features, [#boxes, d]

        saved_dict = {}
        saved_dict['boxes'] = pred_boxes.cpu()
        saved_dict['feats'] = region_feats.cpu()
    else:  # save features of detection regions (after per-class NMS)
        # 4. prediction head classifies the regions (optional)
        predictions = model.roi_heads.box_predictor(
            att_feats)  # predictions[0]: class logits; predictions[1]: box delta
        pred_instances, keep_indices = model.roi_heads.box_predictor.inference(predictions,
                                                                               proposals)  # apply per-class NMS
        results = model._postprocess(pred_instances, batched_inputs)  # re-scale boxes back to original image size

        # save detection outputs into files
        im_id = 0  # single image
        pred_boxes = results[im_id]['instances'].get("pred_boxes").tensor  # boxes after per-class NMS, [#boxes, 4]
        pred_classes = results[im_id]['instances'].get(
            "pred_classes")  # class predictions after per-class NMS, [#boxes], class value in [0, C]
        pred_probs = F.softmax(predictions[0], dim=-1)[
            keep_indices[im_id]]  # class probabilities, [#boxes, #concepts+1], background is the index of C
        region_feats = att_feats[keep_indices[im_id]]  # region features, [#boxes, d]
        # assert torch.all(results[0]['instances'].get("scores") == pred_probs[torch.arange(pred_probs.shape[0]).cuda(), pred_classes]) # scores




        saved_dict = {}
        # saved_dict['boxes'] = pred_boxes.cpu()
        # saved_dict['classes'] = pred_classes.cpu().tolist()
        # saved_dict['probs'] = pred_probs.cpu().tolist()
        # saved_dict['feats'] = region_feats.cpu().tolist()


        filtered_bbox_list = [bbox for bbox, category in zip(pred_boxes.cpu().tolist(), pred_classes.cpu().tolist()) if
                              category == 0]
        saved_dict['boxes'] = filtered_bbox_list[:10]
        saved_dict['name'] = os.path.basename(file_name).split('.')[0]
        saved_dict['class'] = os.path.basename(file_name).split('.')[0].split("_")[0]
        # filtered_probs_list = [pred_prob[0] for pred_prob, category in zip(pred_probs.cpu(), pred_classes.cpu()) if
        #                       category == 0]
        # a = images[0]

        scale_factor = images[0].size()[-2:]
        visualize_bbox(file_name, filtered_bbox_list, cfg.OUTPUT_DIR)



    # pkl_saved_path = os.path.join(cfg.OUTPUT_DIR+"pkl", os.path.basename(file_name).split('.')[0] + '.json')
    # with open(pkl_saved_path, 'w') as f:
    #     json.dump(saved_dict, f)


def main(args):
    cfg = setup(args)

    # create model
    model = create_model(cfg)

    # input images
    image_files = [os.path.join(cfg.INPUT_DIR, x) for x in os.listdir(cfg.INPUT_DIR)]

    # process each image
    start = time.time()
    for i, file_name in tqdm(enumerate(image_files)):
        if i % 100 == 0:
            print("Used {} seconds for 100 images.".format(time.time() - start))
            start = time.time()

        # get input images
        batched_inputs = get_inputs(cfg, file_name)

        # extract region features
        with torch.no_grad():
            extract_region_feats(cfg, model, batched_inputs, file_name)

    print("done!")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
