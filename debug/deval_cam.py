import logging
import multiprocessing as mp
import os
from functools import partial

import hydra
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig


def cal_total_segment_area(masks, segment=0):
    return masks[masks == segment].size + 1e-10


def cal_segment_cam_intersection_area(masks, cam, segment=0):
    mask = masks.copy()
    mask[mask != segment] = 0
    mask = mask * cam
    return mask[mask == segment].size


def apply_cam(segmentation_result, cam_mask, area_threshold):
    segmentation_result += 1
    area_threshold = 1.0 / len(np.unique(cam_mask))
    for i in range(1, np.max(segmentation_result) + 2):
        segment_cam_intersection_area = cal_segment_cam_intersection_area(segmentation_result, cam_mask, segment=i)
        if segment_cam_intersection_area == 0:
            segmentation_result[segmentation_result == i] = 1
            continue
        total_segment_area = cal_total_segment_area(segmentation_result, segment=i)
        area = segment_cam_intersection_area / total_segment_area
        if area > area_threshold:
            segmentation_result[segmentation_result == i] = 2
        else:
            segmentation_result[segmentation_result == i] = 1
        if area > 1:
            print(area)
            raise
    return segmentation_result


def apply_cv_segmentation(segments, high_res_cam, area_threshold, cam_eval_thres):
    cam_mask = np.zeros_like(high_res_cam)
    cam_mask[high_res_cam >= cam_eval_thres] = 1

    return apply_cam(segments.copy(), cam_mask, area_threshold) * high_res_cam


def _work(cam_out_dir, cv_out_dir, cam_eval_thres, area_threshold, idx):
    cam_dict = np.load(os.path.join(cam_out_dir, idx + '.npy'), allow_pickle=True).item()
    cams = cam_dict['high_res']
    if cv_out_dir:
        cv_result = np.load(os.path.join(cv_out_dir, idx + '.npy'), allow_pickle=True).item()
        segments = cv_result['segments']
        cams = np.array([apply_cv_segmentation(segments, lb.copy(), area_threshold, cam_eval_thres) for lb in cams]).astype(
            np.float32)
    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres)
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    cls_labels = np.argmax(cams, axis=0)
    cls_labels = keys[cls_labels]

    return cls_labels


@hydra.main(config_path='../conf', config_name="eval_cam/original")
def run_app(cfg: DictConfig) -> None:
    dataset = VOCSemanticSegmentationDataset(split=cfg.chainer_eval_set, data_dir=cfg.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    debug = True
    if debug:
        preds = []
        for idx in dataset.ids:
            pred = _work(cfg.cam_out_dir, cfg.cv_out_dir, cfg.cam_eval_thres, cfg.area_threshold, idx)
            preds.append(pred)
    else:
        with mp.Pool(processes=mp.cpu_count() // 2) as pool:
            preds = pool.map(partial(_work, cfg.cam_out_dir, cfg.cv_out_dir, cfg.cam_eval_thres, cfg.area_threshold),
                             list(dataset.ids))
    print(len(preds))

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})
    logging.info({'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':
    run_app()
