import importlib
import os

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from irn.voc12 import dataloader


def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays

    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores


@hydra.main(config_path='../conf', config_name="make_cam")
def run_app(cfg: DictConfig) -> None:
    model = getattr(importlib.import_module(cfg.cam_network), 'Net')()
    model.load_state_dict(torch.load(cfg.cam_weights_name + '.pth', map_location=torch.device('cpu')), strict=True)
    model.eval()
    os.makedirs(cfg.cam_out_dir, exist_ok=True)

    dataset = dataloader.VOC12ClassificationDataset(cfg.train_list, voc12_root=cfg.voc12_root, crop_size=512)

    data_loader = DataLoader(dataset, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, batch_size=8)
    running_ap = 0
    with torch.no_grad():
        for iter, pack in enumerate(data_loader):
            label = pack['label']

            outputs = model(pack['img'])
            outputs = torch.sigmoid(outputs)
            running_ap += get_ap_score(label.cpu().detach().numpy(),
                                       outputs.cpu().detach().numpy())
            print(iter)
    print(running_ap / len(dataset))


if __name__ == "__main__":
    run_app()
