import importlib
import os

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from irn.misc import indexing
from irn.voc12 import dataloader


@hydra.main(config_path='../conf', config_name="make_sem_seg_labels")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.sem_seg_out_dir, exist_ok=True)
    model = getattr(importlib.import_module(cfg.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(cfg.irn_weights_name), strict=False)
    model.eval()

    dataset = dataloader.VOC12ClassificationDatasetMSF(cfg.infer_list,
                                                       voc12_root=cfg.voc12_root,
                                                       scales=(1.0,))

    data_loader = DataLoader(dataset,
                             shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

    with torch.no_grad():
        for iter, pack in enumerate(data_loader):
            img_name = dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0])

            cam_dict = np.load(cfg.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=cfg.beta, exp_times=cfg.exp_times,
                                            radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0],
                    :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=cfg.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(cfg.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            print(iter)


if __name__ == "__main__":
    run_app()
