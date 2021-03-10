import importlib

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from irn.misc import pyutils, torchutils
from irn.voc12 import dataloader


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label']

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


@hydra.main(config_path='../conf', config_name="train_cam")
def run_app(cfg: DictConfig) -> None:
    model = getattr(importlib.import_module(cfg.cam_network), 'Net')()

    if cfg.weights:
        try:
            weights_dict = torch.load(cfg.weights, map_location=torch.device('cpu'))
            model.load_state_dict(weights_dict, strict=False)
        except:
            pass

    train_dataset = dataloader.VOC12ClassificationDataset(cfg.train_list, voc12_root=cfg.voc12_root,
                                                          resize_long=(320, 640), hor_flip=True,
                                                          crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.cam_batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // cfg.cam_batch_size) * cfg.cam_num_epoches

    val_dataset = dataloader.VOC12ClassificationDataset(cfg.val_list, voc12_root=cfg.voc12_root,
                                                        crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg.cam_batch_size,
                                 shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cfg.cam_learning_rate, 'weight_decay': cfg.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * cfg.cam_learning_rate, 'weight_decay': cfg.cam_weight_decay},
    ], lr=cfg.cam_learning_rate, weight_decay=cfg.cam_weight_decay, max_step=max_step)

    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(cfg.cam_num_epoches):

        print('Epoch %d/%d' % (ep + 1, cfg.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label']

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * cfg.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.state_dict(), cfg.cam_weights_name + '.pth')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_app()
