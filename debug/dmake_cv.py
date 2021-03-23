import multiprocessing as mp
import os
from functools import partial

import hydra
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from omegaconf import DictConfig
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


def apply_cv_segmentation(img, cv_method):
    segments = None
    if cv_method == 'felzenszwalb':
        segments = felzenszwalb(img, scale=100, sigma=0.9, min_size=10)
    if cv_method == 'slic':
        segments = slic(img, n_segments=100, compactness=10, sigma=1, start_label=1)
    if cv_method == 'quickshift':
        segments = quickshift(img.astype(np.double), kernel_size=3, max_dist=6, ratio=0.5)
    if cv_method == 'watershed':
        gradient = sobel(rgb2gray(img))
        segments = watershed(gradient, markers=250, compactness=0.001)

    return segments


def _work(dataset, out_dir, cv_method, i, idx):
    img, _ = dataset.get_example(i)
    img = img.transpose(1, 2, 0)
    segments = apply_cv_segmentation(img, cv_method)

    np.save(os.path.join(out_dir, idx + '.npy'), {"segments": segments})
    return True


@hydra.main(config_path='../conf', config_name="make_cv/slic")
def run_app(cfg: DictConfig) -> None:
    dataset = VOCSemanticSegmentationDataset(split=cfg.chainer_eval_set, data_dir=cfg.voc12_root)
    os.makedirs(cfg.out_dir, exist_ok=True)
    debug = False
    if debug:
        for i, idx in enumerate(dataset.ids):
            _work(dataset, cfg.out_dir, cfg.cv_method, i, idx)

    else:
        with mp.Pool(processes=mp.cpu_count() // 2) as pool:
            pool.starmap(partial(_work, dataset, cfg.out_dir, cfg.cv_method), enumerate(dataset.ids))


if __name__ == '__main__':
    run_app()
