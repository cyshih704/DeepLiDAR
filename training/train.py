from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from training.utils import get_loss



def get_optimizer(model, stage):
    """(1) Get optimizer at different stage.
       (2) Set require_grad of parameters at different stage
       (3) Get weights of each loss at different stage
       For example, at stage N, we only need to train surface normal.

       Returns: 
       model
       optimizer
       loss: list, weight of loss_c, loss_d, loss, loss_normal
    """
    assert stage in {'D', 'N', 'A'}

    if stage == 'N':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.normal.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.normal.parameters(), lr=0.001, betas=(0.9, 0.999))
        loss_weights = [0, 0, 0, 1]

    elif stage == 'D':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.color_path.parameters():
            param.requires_grad = True
        for param in model.normal_path.parameters():
            param.requires_grad = True

        optimizer = optim.Adam([{'params':model.color_path.parameters()},
                                {'params':model.normal_path.parameters()}], lr=0.001, betas=(0.9, 0.999))
        loss_weights = [0.3, 0.3, 0.0, 0.1]

    else:
        for param in model.color_path.parameters():
            param.requires_grad = True  
        for param in model.normal.parameters():
            param.requires_grad = False  

        optimizer = optim.Adam([{'params':model.color_path.parameters()},
                                {'params':model.normal_path.parameters()},
                                {'params':model.mask_block_C.parameters()},
                                {'params':model.mask_block_N.parameters()}], lr=0.001, betas=(0.9, 0.999))

        loss_weights = [0.3, 0.3, 0.5, 0.1]

    return model, optimizer, loss_weights



def train_val(model, loader, epoch, device, stage):
    """Train and validate the model

    Returns: training and validation loss
    """

    model, optimizer, loss_weights = get_optimizer(model, stage)
    train_loss, val_loss = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

    for phase in ['train', 'val']:
        total_loss, total_loss_d, total_loss_c, total_loss_n, total_loss_normal = 0, 0, 0, 0, 0
        total_pic = 0 # used to calculate average loss
        data_loader = loader[phase]
        pbar = tqdm(iter(data_loader))

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for num_batch, (rgb, lidar, mask, gt_depth, params, gt_surface_normal, gt_normal_mask) in enumerate(pbar):
            """
            rgb: b x 3 x 128 x 256
            lidar: b x 1 x 128 x 256
            mask: b x 1 x 128 x 256
            gt: b x 1 x 128 x 256
            params: b x 128 x 256 x 3
            """
            rgb, lidar, mask = rgb.to(device), lidar.to(device), mask.to(device)
            gt_depth, params = gt_depth.to(device), params.to(device)
            gt_surface_normal, gt_normal_mask = gt_surface_normal.to(device), gt_normal_mask.to(device)

            if phase == 'train':
                color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal = model(rgb, lidar, mask, stage)
            else:
                with torch.no_grad():
                    color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal = model(rgb, lidar, mask, stage)
            # color_path_dense: b x 2 x 128 x 256
            # normal_path_dense: b x 2 x 128 x 256
            # color_mask: b x 1 x 128 x 256
            # normal_mask: b x 1 x 128 x 256
            # surface_normal: b x 3 x 128 x 256

            loss_c, loss_n, loss_d, loss_normal = get_loss(color_path_dense, normal_path_dense, color_attn,\
                                                            normal_attn, pred_surface_normal, stage,\
                                                            gt_depth, params, gt_surface_normal, gt_normal_mask)

            loss = loss_weights[0] * loss_c + loss_weights[1] * loss_n + loss_weights[2] * loss_d + loss_weights[3] * loss_normal

            total_loss += loss.item()
            total_loss_d += loss_d.item()
            total_loss_c += loss_c.item()
            total_loss_n += loss_n.item()
            total_loss_normal += loss_normal.item()

            total_pic += rgb.size(0)

            if phase == 'train':
                train_loss[0] = total_loss/total_pic
                train_loss[1] = total_loss_d/total_pic
                train_loss[2] = total_loss_c/total_pic
                train_loss[3] = total_loss_n/total_pic
                train_loss[4] = total_loss_normal/total_pic

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            else:
                val_loss[0] = total_loss/total_pic
                val_loss[1] = total_loss_d/total_pic
                val_loss[2] = total_loss_c/total_pic
                val_loss[3] = total_loss_n/total_pic
                val_loss[4] = total_loss_normal/total_pic

            pbar.set_description('[{}] Epoch: {}; loss: {:.4f}; loss_d: {:.4f}, loss_c: {:.4f}, loss_n: {:.4f}, loss_normal: {:.4f}'.\
                format(phase.upper(), epoch + 1, total_loss/total_pic , total_loss_d/total_pic, \
                total_loss_c/total_pic, total_loss_n/total_pic, total_loss_normal/total_pic))

    return train_loss, val_loss

class EarlyStop():
    """Early stop training if validation loss didn't improve for a long time"""
    def __init__(self, patience, mode = 'min'):
        self.patience = patience
        self.mode = mode

        self.best = float('inf') if mode == 'min' else 0
        self.cur_patience = 0

    def stop(self, loss, model, epoch, saved_model_path):
        update_best = loss < self.best if self.mode == 'min' else loss > self.best

        if update_best:
            self.best = loss
            self.cur_patience = 0

            torch.save({'val_loss': loss, \
                        'state_dict': model.state_dict(), \
                        'epoch': epoch}, saved_model_path+'.tar')
            print('SAVE MODEL to {}'.format(saved_model_path))
        else:
            self.cur_patience += 1
            if self.patience == self.cur_patience:
                return True
        
        return False