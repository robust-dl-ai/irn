import os

import hydra
import imageio
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig

from irn.debug.deval_cam import cal_segment_cam_intersection_area, cal_total_segment_area


def add_cv_results(cls_labels, idx, cv_out_dir, area_threshold):
    if cv_out_dir:
        cv_result = np.load(os.path.join(cv_out_dir, idx + '.npy'), allow_pickle=True).item()
        segments = cv_result['segments']
        segments += 1

        for cls_lbl in np.unique(cls_labels):
            mask = np.zeros_like(cls_labels)
            if cls_lbl == 0:
                continue
            mask[cls_labels == cls_lbl] = 1
            for i in range(np.max(segments) + 1):
                segment_cam_intersection_area = cal_segment_cam_intersection_area(segments, mask, segment=i)
                total_segment_area = cal_total_segment_area(segments, segment=i)
                area = segment_cam_intersection_area / total_segment_area
                if area > area_threshold:
                    cls_labels[segments == i] = cls_lbl

    return cls_labels


@hydra.main(config_path='../conf', config_name="eval_sem/watershed")
def run_app(cfg: DictConfig) -> None:
    dataset = VOCSemanticSegmentationDataset(split=cfg.chainer_eval_set, data_dir=cfg.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    for id in dataset.ids:
        cls_labels = imageio.imread(os.path.join(cfg.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        if cfg.cv_out_dir:
            cls_labels = add_cv_results(cls_labels.copy(), id, cfg.cv_out_dir, cfg.area_threshold)
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':
    run_app()
