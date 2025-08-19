"""
    This script contains function snippets for different training settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from visualDet3D.utils.utils import LossLogger
from visualDet3D.utils.utils import compound_annotation, compound_annotation_2d_only
from visualDet3D.networks.utils.registry import PIPELINE_DICT


@PIPELINE_DICT.register_module
def train_mono_detection(data, module:nn.Module,
                     optimizer:optim.Optimizer,
                     writer:SummaryWriter=None, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     epoch_num:int=None,
                     cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    # load data
    image, calibs, labels, bbox2d, bbox_3d = data

    # create compound array of annotation
    max_length = np.max([len(label) for label in labels])
    if max_length == 0:
        return
    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types) #np.arraym, [batch, max_length, 4 + 1 + 7]

    # Feed to the network
    classification_loss, fg_loss, regression_loss, loss_dict = module(
            [image.cuda().contiguous(), image.new(annotation).cuda(), calibs.cuda()])

    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()
    fg_loss = fg_loss.mean()

    # Record loss in a average meter
    if loss_logger is not None:
        loss_logger.update(loss_dict)

    loss = classification_loss + regression_loss + fg_loss

    if bool(loss.item() == 0):
        return
    loss.backward()
    # clip loss norm
    torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)

    optimizer.step()
    optimizer.zero_grad()

@PIPELINE_DICT.register_module
def train_mono_depth(data, module:nn.Module,
                     optimizer:optim.Optimizer,
                     writer:SummaryWriter=None, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     epoch_num:int=None,
                     cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    image, K, gts = data

    # Feed to the network
    loss, loss_dict = module(
            [image.cuda().float().contiguous(), image.new(K).cuda().float(), gts.cuda().float()]
        )

    if not loss_logger is None and loss > 0:
        # Record loss in a average meter
        loss_logger.update(loss_dict)

    if bool(loss == 0):
        return
    loss.backward()
    # clip loss norm
    torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)

    optimizer.step()

@PIPELINE_DICT.register_module
def train_stereo_detection(data, module:nn.Module,
                     optimizer:optim.Optimizer,
                     writer:SummaryWriter=None, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None,
                     epoch_num:int=None,
                     cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    left_images, right_images, P2, P3, labels, bbox2d, bbox_3d, disparity, label_ext, bbox2ds_ext = data
    # create compound array of annotation
    max_length = np.max([len(label) for label in labels+label_ext])
    if max_length == 0:
       return
    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types) #np.arraym, [batch, max_length, 4 + 1 + 7]
    annotation_ext = compound_annotation_2d_only(label_ext, max_length, bbox2ds_ext, cfg.obj_types_fg) #np.arraym, [batch, max_length, 4 + 1]
    # Feed to the network
    classification_loss, foreground_loss, regression_loss, loss_dict = module(
            [left_images.cuda().float().contiguous(), right_images.cuda().float().contiguous(),
             left_images.new(annotation).cuda(),
             left_images.new(annotation_ext).cuda(),
             P2.cuda(), P3.cuda(),
             disparity.cuda().contiguous()]
        )

    classification_loss = classification_loss.mean()
    foreground_loss = foreground_loss.mean()
    regression_loss = regression_loss.mean()

    if not loss_logger is None:
        # Record loss in a average meter
        loss_logger.update(loss_dict)
    del loss_dict

    if not optimizer is None:
        loss = classification_loss + foreground_loss + regression_loss

    if bool(loss == 0):
        del loss, loss_dict
        return
    loss.backward()
    # clip loss norm
    # 检查梯度是否有限
    if torch.isfinite(loss).all():
        # 如果梯度有限，则裁剪梯度
        torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)
        optimizer.step()
    else:
        # 如果梯度非有限，则跳过裁剪
        print("Warning: Non-finite gradients detected. Skipping gradient clipping.")
    # with torch.autograd.detect_anomaly():
    #     torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm,
    #                                error_if_nonfinite=True)
    
    optimizer.zero_grad()

@PIPELINE_DICT.register_module
def train_rtm3d(data, module:nn.Module,
                     optimizer:optim.Optimizer,
                     writer:SummaryWriter=None, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None,
                     epoch_num:int=None,
                     cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    image, K, gts = data
    #outs = data
    
    for key in gts:
        gts[key] = gts[key].cuda()
    
    # Feed to the network
    loss, loss_dict = module(
            [image.cuda().float().contiguous(), gts, dict(P2=image.new(K).cuda().float(), epoch=epoch_num)]
        )

    
    if not loss_logger is None and loss > 0:
        # Record loss in a average meter
        loss_logger.update(loss_dict)

    if bool(loss == 0):
        return
    loss.mean().backward()
    # clip loss norm
    if 'clipped_gradient_norm' in cfg.optimizer:
        torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)
    optimizer.step()
