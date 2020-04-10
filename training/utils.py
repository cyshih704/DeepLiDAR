import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn):
    """Use raw model output to generate dense of color pathway, normal path way, and integrated result

    Returns: predicted_dense, pred_color_path_dense, pred_normal_path_dense
    """
    # get predicted dense depth from 2 pathways
    pred_color_path_dense = color_path_dense[:, 0, :, :] # b x 128 x 256
    pred_normal_path_dense = normal_path_dense[:, 0, :, :]

    # get attention map of 2 pathways
    color_attn = torch.squeeze(color_attn) # b x 128 x 256
    normal_attn = torch.squeeze(normal_attn) # b x 128 x 256

    # softmax 2 attention map
    pred_attn = torch.zeros_like(color_path_dense) # b x 2 x 128 x 256
    pred_attn[:, 0, :, :] = color_attn
    pred_attn[:, 1, :, :] = normal_attn
    pred_attn = F.softmax(pred_attn, dim=1) # b x 2 x 128 x 256

    color_attn, normal_attn = pred_attn[:, 0, :, :], pred_attn[:, 1, :, :]

    # get predicted dense from weighted sum of 2 path way
    predicted_dense = pred_color_path_dense * color_attn + pred_normal_path_dense * normal_attn # b x 128 x 256

    predicted_dense = predicted_dense.unsqueeze(1)
    pred_color_path_dense = pred_color_path_dense.unsqueeze(1) 
    pred_normal_path_dense = pred_normal_path_dense.unsqueeze(1)

    return predicted_dense, pred_color_path_dense, pred_normal_path_dense

def get_depth_and_normal(model, rgb, lidar, mask):
    """Given model and input of model, get dense depth and surface normal

    Returns:
    predicted_dense: b x c x h x w
    pred_surface_normal: b x c x h x w
    """
    model.eval()
    with torch.no_grad():
        color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal = model(rgb, lidar, mask, 'A')
        predicted_dense, _, _ = get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)
    return predicted_dense, pred_surface_normal




def normal_to_0_1(img):
    """Normalize image to [0, 1], used for tensorboard visualization."""
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def normal_loss(pred_normal, gt_normal, gt_normal_mask):
    """Calculate loss of surface normal (in the stage N)

    Params:
    pred: b x 3 x 128 x 256
    normal_gt: b x 3 x 128 x 256
    normal_mask: b x 3 x 128 x 256
    """

    valid_mask = (gt_normal_mask > 0.0).detach()

    #pred_n = pred.permute(0,2,3,1)
    pred_normal = pred_normal[valid_mask]
    gt_normal = gt_normal[valid_mask]

    pred_normal = pred_normal.contiguous().view(-1,3)
    pred_normal = F.normalize(pred_normal)
    gt_normal = gt_normal.contiguous().view(-1, 3)

    loss_function = nn.CosineEmbeddingLoss()
    loss = loss_function(pred_normal, gt_normal, torch.Tensor(pred_normal.size(0)).to(pred_normal.device).fill_(1.0))
    return loss



def get_depth_loss(dense, c_dense, n_dense, gt):
    """
    dense: b x 1 x 128 x 256
    c_dense: b x 1 x 128 x 256
    n_dense: b x 1 x 128 x 256
    gt: b x 1 x 128 x 256
    params: b x 3 x 128 x 256
    normals: b x 128 x 256 x 3
    """
    valid_mask = (gt > 0.0).detach() # b x 1 x 128 x 256

    gt = gt[valid_mask]
    dense, c_dense, n_dense = dense[valid_mask], c_dense[valid_mask], n_dense[valid_mask]

    criterion = nn.MSELoss()
    loss_d = torch.sqrt(criterion(dense, gt))
    loss_c = torch.sqrt(criterion(c_dense, gt))
    loss_n = torch.sqrt(criterion(n_dense, gt))
    
    return loss_d, loss_c, loss_n




def get_loss(color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal, stage, gt_depth, params, gt_surface_normal, gt_normal_mask):
    assert stage in {'D', 'N', 'A'}

    zero_loss = nn.MSELoss()(torch.ones(1, 1).to(gt_depth.device), torch.ones(1, 1).to(gt_depth.device))
    loss_d, loss_c, loss_n, loss_normal = zero_loss, zero_loss, zero_loss, zero_loss

    if stage == 'N':
        loss_normal = normal_loss(pred_surface_normal, gt_surface_normal, gt_normal_mask)
    else:
 
        predicted_dense, pred_color_path_dense, pred_normal_path_dense = \
                            get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)

        # normalize surface normal
        #b, c, h, w = pred_surface_normal.size()
        #pred_surface_normal = pred_surface_normal.permute(0, 2, 3, 1).contiguous().view(-1, c)
        #pred_surface_normal = F.normalize(pred_surface_normal, p=2, dim=1) # perform Lp normalization over specific dimension
        #pred_surface_normal = pred_surface_normal.view(b, h, w, c)
        ## TODO
        #output_normal = torch.zeros_like(pred_surface_normal)
        #output_normal[:, :, :, 0] = -pred_surface_normal[:, :, :, 0]
        #output_normal[:, :, :, 1] = -pred_surface_normal[:, :, :, 2]
        #output_normal[:, :, :, 2] = -pred_surface_normal[:, :, :, 1]

        loss_d, loss_c, loss_n = get_depth_loss(predicted_dense, pred_color_path_dense, pred_normal_path_dense, gt_depth)
        loss_normal = normal_loss(pred_surface_normal, gt_surface_normal, gt_normal_mask)

    return loss_c, loss_n, loss_d, loss_normal
