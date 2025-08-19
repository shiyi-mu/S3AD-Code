import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.lib.ghost_module import ResGhostModule, GhostModule
from visualDet3D.networks.lib.PSM_cost_volume import PSMCosineModule, CostVolume
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.look_ground import LookGround

class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost

class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4  = depth_channel_4 # 24
        self.depth_channel_8  = depth_channel_8 # 24
        self.depth_channel_16 = depth_channel_16 # 96

        input_features = depth_channel_4 # 24
        self.four_to_eight = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        self.output_channel_num = 3 * input_features #1152

        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )


    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4_8 = self.four_to_eight(psv_volume_4)
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1)
        psv_16 = self.depth_reason(psv_volume_16)
        if self.training:
            return psv_16, self.depth_output(psv_16)
        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])

class StereoMerging(nn.Module):
    def __init__(self, base_features):
        super(StereoMerging, self).__init__()
        self.cost_volume_0 = PSMCosineModule(downsample_scale=4, max_disp=96, input_features=base_features)
        PSV_depth_0 = self.cost_volume_0.depth_channel

        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=192, input_features=base_features * 2)
        PSV_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=192, input_features=base_features * 4, PSM_features=8)
        PSV_depth_2 = self.cost_volume_2.output_channel

        self.depth_reasoning = CostVolumePyramid(PSV_depth_0, PSV_depth_1, PSV_depth_2)
        self.final_channel = self.depth_reasoning.output_channel_num + base_features * 4

    def forward(self, left_x, right_x):
        PSVolume_0 = self.cost_volume_0(left_x[0], right_x[0])
        PSVolume_1 = self.cost_volume_1(left_x[1], right_x[1])
        PSVolume_2 = self.cost_volume_2(left_x[2], right_x[2])
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2) # c = 1152
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 1152 + 256 = 1408
        return features, depth_output, PSV_features


class Aggregation(nn.Module):
    def __init__(self, in_channels, in_channels_s8, in_channels_s16, left_att, blocks, expanse_ratio, backbone_channels, cat_left=False):
        super(Aggregation, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio[0]
        self.expanse_ratio_8 = expanse_ratio[1]
        self.expanse_ratio_16 = expanse_ratio[2]
        self.cat_left = cat_left

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        conv0_8 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio_8)
                 for i in range(blocks[0])]
        self.conv0_8 = nn.Sequential(*conv0_8)

        conv0_16 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio_16)
                 for i in range(blocks[0])]
        self.conv0_16 = nn.Sequential(*conv0_16)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [MobileV2Residual(in_channels * 2 + in_channels_s8, 
                                      in_channels * 2 + in_channels_s8, 
                                      stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = MobileV2Residual(in_channels * 2 + in_channels_s8, 
                                      in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [MobileV2Residual(in_channels * 4 + in_channels_s16, 
                                      in_channels * 4 + in_channels_s16, 
                                      stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2 + in_channels_s8, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4 + in_channels_s16, backbone_channels[2])
        
        if not self.cat_left:   
            input_features = in_channels * 4 + in_channels_s16 
            self.depth_reason = nn.Sequential(
                ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
                MobileV2Residual(3 * input_features, 
                                512, stride=1, expanse_ratio=self.expanse_ratio)
            )
        else:
            input_features = in_channels * 4 + in_channels_s16 + 256
            self.depth_reason = nn.Sequential(
                ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
                MobileV2Residual(3 * input_features, 
                                512, stride=1, expanse_ratio=self.expanse_ratio)
            )
    def forward(self, x, x_8, x_16, features_left):

        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])
        conv1 = self.conv1(x) # b, 48, 36, 160
       
        conv1 = torch.cat([x_8, conv1], dim=1) # b, 48+24, 36, 160
        conv2 = self.conv2(conv1)
    
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])
        conv3 = self.conv3(conv2)
        
        conv3 = torch.cat([x_16, conv3], dim=1) # b, 48+24+12, 36, 160
      
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])
        if not self.cat_left:   
            conv5 = self.depth_reason(conv4)
        else:
            conv5 = self.depth_reason(torch.cat([conv4, features_left[2]], dim=1))
        return conv5


class StereoMergingLight(nn.Module):
    def __init__(self, base_features):
        super(StereoMergingLight, self).__init__()
        # aggregation
        self.cost_agg = Aggregation(in_channels=24,
                                    in_channels_s8=24, #24
                                    in_channels_s16=12,
                                    left_att=True,
                                    blocks=[ 1, 2, 4 ],
                                    expanse_ratio=[4, 4, 4],
                                    backbone_channels=[ 64, 128, 256 ],
                                    cat_left=False)
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=256),
            nn.ReLU())
        self.output_channel_num = 256
        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )
    def correlation_volume(self, left_feature, right_feature, max_disp):
        b, c, h, w = left_feature.size()
        cost_volume = left_feature.new_zeros(b, max_disp, h, w)
        for i in range(max_disp):
            if i > 0:
                cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
            else:
                cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        cost_volume = cost_volume.contiguous()
        return cost_volume
    
    def forward(self, left_x, right_x):
        gwc_volume_s4 = self.correlation_volume(left_x[0], 
                                               right_x[0], 
                                               96 // 4)
        gwc_volume_s8 = self.correlation_volume(left_x[1], 
                                            right_x[1], 
                                            192 // 8)  # 192/8
        gwc_volume_s16 = self.correlation_volume(left_x[2], 
                                            right_x[2], 
                                            192 // 16)
        features_left = left_x
        PSV_features = self.cost_agg(gwc_volume_s4, gwc_volume_s8, gwc_volume_s16, features_left)
        src = self.depth_head(PSV_features)
        depth_output = self.depth_output(src)
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 512 + 256 = 768
        return features, depth_output, PSV_features

class StereoMerging2Mono(nn.Module):
    def __init__(self, base_features):
        super(StereoMerging2Mono, self).__init__()
        self.depth_reasoning = CostVolumePyramid(24, 24, 96)
        # 1*1卷积，大小不变仅变换通道
        self.cost_volume_0 = nn.Conv2d(64, 24, 1, bias=False)

        self.cost_volume_1 = nn.Conv2d(128, 24, 1, bias=False)

        self.cost_volume_2 = nn.Conv2d(256, 96, 1, bias=False)

    def forward(self, left_x):
        PSVolume_0 = self.cost_volume_0(left_x[0])
        PSVolume_1 = self.cost_volume_1(left_x[1])
        PSVolume_2 = self.cost_volume_2(left_x[2])
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2) # c = 1152
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 1152 + 256 = 1408
        return features, depth_output, PSV_features

class YoloStereo3DCore(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore, self).__init__()
        self.backbone =resnet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging(base_features)


    def forward(self, images):

        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]

        images = torch.cat([left_images, right_images], dim=0)

        features = self.backbone(images)

        left_features  = [feature[0:batch_size] for feature in features]
        right_features = [feature[batch_size:]  for feature in features]

        features, depth_output, PSV_features = self.neck(left_features, right_features)
        left_features_final = left_features[2]
        output_dict = dict(features=features, 
                           depth_output=depth_output, 
                           PSV_features=PSV_features,
                           left_features_final=left_features_final)
        return output_dict 


class YoloStereo3DCore2Mono(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore2Mono, self).__init__()
        self.backbone =resnet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging2Mono(base_features)


    def forward(self, images):

        batch_size = images.shape[0]
        left_images = images
        left_features = self.backbone(left_images)

        features, depth_output, PSV_features = self.neck(left_features)

        output_dict = dict(features=features, depth_output=depth_output, PSV_features=PSV_features)
        return output_dict

class YoloStereo3DLightCore(YoloStereo3DCore):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DLightCore, self).__init__(backbone_arguments)
        self.backbone =resnet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMergingLight(base_features)
