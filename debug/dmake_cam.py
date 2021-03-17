import importlib
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from irn.misc import imutils
from irn.voc12 import dataloader


@hydra.main(config_path='../conf', config_name="make_cam")
def run_app(cfg: DictConfig) -> None:
    model = getattr(importlib.import_module(cfg.cam_network), 'CAM')()
    model.load_state_dict(torch.load(cfg.cam_weights_name + '.pth', map_location=torch.device('cpu')), strict=True)
    model.eval()
    os.makedirs(cfg.cam_out_dir, exist_ok=True)

    dataset = dataloader.VOC12ClassificationDatasetMSF(cfg.train_list,
                                                       voc12_root=cfg.voc12_root, scales=cfg.cam_scales)

    data_loader = DataLoader(dataset, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

    with torch.no_grad():
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0]) for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(cfg.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
            print(iter)


if __name__ == "__main__":
    run_app()
