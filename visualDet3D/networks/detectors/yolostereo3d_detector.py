import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.utils.timer import profile
from visualDet3D.networks.heads import losses
from visualDet3D.networks.detectors.yolostereo3d_core import YoloStereo3DCore,YoloStereo3DLightCore, YoloStereo3DCore2Mono
from visualDet3D.networks.heads.detection_3d_head import StereoHead, StereoHead_Light, StereoHead_fgByLeftStereo, StereoHead_fgByLeft
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.backbones.resnet import BasicBlock



@DETECTOR_DICT.register_module
class Stereo3D(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, annotations_ext, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            annotations_ext: all 2Dbboxes (include some box without 3D)
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    left_features_final=output_dict['left_features_final'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)
        cls_loss_know, _, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, 
                                                        reg_preds, 
                                                        anchors, 
                                                        annotations, 
                                                        P2)
        
        cls_loss_ext, fg_loss, reg_loss_ext, loss_dict_ext = self.bbox_head.loss_ext(cls_preds, 
                                                                     reg_preds, 
                                                                     anchors, 
                                                                     annotations_ext,
                                                                     P2)
        reg_loss += reg_loss_ext
        cls_loss = cls_loss_ext
        total_loss = loss_dict_ext["total_loss"] + loss_dict["total_loss"]
        loss_dict = dict(cls_loss=cls_loss, 
                            fg_loss=fg_loss, 
                            reg_loss=reg_loss, 
                            total_loss=total_loss)
        
        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, fg_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    left_features_final=output_dict['left_features_final'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes, ls_score_backup, anchor_backup = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        return scores, bboxes, cls_indexes, ls_score_backup, anchor_backup


    def forward(self, inputs):
        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)


@DETECTOR_DICT.register_module
class Stereo3DFgByLeftStereo(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3DFgByLeftStereo, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead_fgByLeftStereo(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, annotations_ext, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            annotations_ext: all 2Dbboxes (include some box without 3D)
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)
        cls_loss_know, _, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, 
                                                        reg_preds, 
                                                        anchors, 
                                                        annotations, 
                                                        P2)
        
        cls_loss_ext, fg_loss, reg_loss_ext, loss_dict_ext = self.bbox_head.loss_ext(cls_preds, 
                                                                     reg_preds, 
                                                                     anchors, 
                                                                     annotations_ext,
                                                                     P2)
        reg_loss += reg_loss_ext
        cls_loss = cls_loss_ext
        total_loss = loss_dict_ext["total_loss"] + loss_dict["total_loss"]
        loss_dict = dict(cls_loss=cls_loss, 
                            fg_loss=fg_loss, 
                            reg_loss=reg_loss, 
                            total_loss=total_loss)
        
        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, fg_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes, ls_score_backup, anchor_backup = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        return scores, bboxes, cls_indexes, ls_score_backup, anchor_backup


    def forward(self, inputs):
        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)

@DETECTOR_DICT.register_module
class Stereo3DFgByLeft(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3DFgByLeft, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead_fgByLeft(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, annotations_ext, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            annotations_ext: all 2Dbboxes (include some box without 3D)
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    left_features_final=output_dict['left_features_final'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)
        cls_loss_know, _, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, 
                                                        reg_preds, 
                                                        anchors, 
                                                        annotations, 
                                                        P2)
        
        cls_loss_ext, fg_loss, reg_loss_ext, loss_dict_ext = self.bbox_head.loss_ext(cls_preds, 
                                                                     reg_preds, 
                                                                     anchors, 
                                                                     annotations_ext,
                                                                     P2)
        reg_loss += reg_loss_ext
        cls_loss = cls_loss_ext
        total_loss = loss_dict_ext["total_loss"] + loss_dict["total_loss"]
        loss_dict = dict(cls_loss=cls_loss, 
                            fg_loss=fg_loss, 
                            reg_loss=reg_loss, 
                            total_loss=total_loss)
        
        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, fg_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    left_features_final=output_dict['left_features_final'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes, ls_score_backup, anchor_backup = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        return scores, bboxes, cls_indexes, ls_score_backup, anchor_backup


    def forward(self, inputs):
        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)
               
@DETECTOR_DICT.register_module
class Stereo3D2Mono(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3D2Mono, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore2Mono(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, annotations_ext, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            annotations_ext: all 2Dbboxes (include some box without 3D)
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(left_images)
        depth_output = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)
        cls_loss_know, _, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, 
                                                        reg_preds, 
                                                        anchors, 
                                                        annotations, 
                                                        P2)
        
        cls_loss_ext, fg_loss, reg_loss_ext, loss_dict_ext = self.bbox_head.loss_ext(cls_preds, 
                                                                     reg_preds, 
                                                                     anchors, 
                                                                     annotations_ext,
                                                                     P2)
        reg_loss += reg_loss_ext
        cls_loss = cls_loss_ext
        total_loss = loss_dict_ext["total_loss"] + loss_dict["total_loss"]
        loss_dict = dict(cls_loss=cls_loss, 
                            fg_loss=fg_loss, 
                            reg_loss=reg_loss, 
                            total_loss=total_loss)
        
        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, fg_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(left_images)
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    PSV_features=output_dict['PSV_features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes, ls_score_backup, anchor_backup = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        return scores, bboxes, cls_indexes, ls_score_backup, anchor_backup


    def forward(self, inputs):
        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)

@DETECTOR_DICT.register_module
class Stereo3DLight(Stereo3D):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3DLight, self).__init__(network_cfg)

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DLightCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead_Light(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    