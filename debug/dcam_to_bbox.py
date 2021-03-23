import os

import cv2
import hydra
import imageio
import imutils as im_utils
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from irn.misc import imutils
from irn.voc12 import dataloader


def generate_bbox(mask_img):
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=8)

    cnts, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = None, None, None, None
    cnts = im_utils.grab_contours((cnts, hierarchy))
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(mask_img,[box],0,(0,0,255),2)
        x, y, w, h = cv2.boundingRect(c)

    return x, y, w, h


@hydra.main(config_path='../conf', config_name="cam_to_ir_label")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.ir_label_out_dir, exist_ok=True)
    dataset = dataloader.VOC12ImageDataset(cfg.train_list, voc12_root=cfg.voc12_root, img_normal=None,
                                           to_torch=False)

    infer_data_loader = DataLoader(dataset, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(cfg.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cfg.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cfg.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        for cls in np.unique(conf):
            if cls != 0:
                pred_a = conf.copy()
                pred_a[pred_a != cls] = 0
                pred_a[pred_a == cls] = 255
                x, y, w, h = generate_bbox(pred_a)
                print(y, x, h, w, img_name, cls)

        imageio.imwrite(os.path.join(cfg.ir_label_out_dir, img_name + '.png'),
                        conf.astype(np.uint8))

        print(iter)


if __name__ == "__main__":
    run_app()
