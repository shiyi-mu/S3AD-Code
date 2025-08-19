import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision
import math
import time
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.depth_anything_v2 import DINOv2
from visualDet3D.networks.depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch



def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        # print("layer_1", layer_1.shape)
        # print("layer_2", layer_2.shape)
        # print("layer_3", layer_3.shape)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        f_output_conv1 = self.scratch.output_conv1(path_1)

        out = F.interpolate(f_output_conv1, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        depth_f = F.interpolate(f_output_conv1, (18, 80), mode="bilinear", align_corners=True)
        return out, depth_f

class YoloMono3DCoreDA(nn.Module):
    """Some Information about YoloMono3DCore"""
    def __init__(self, backbone_arguments=dict()):
        super(YoloMono3DCoreDA, self).__init__()

        # encoder='vits'
        # features=64
        # out_channels= [48, 96, 192, 384]
        encoder = 'vitb'
        features= 128
        out_channels= [96, 192, 384, 768]

        use_bn=False
        use_clstoken=False

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.encoder = encoder
        self.backbone =resnet(**backbone_arguments)
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        # 从depth anythinv2 load模型参数：
        depth_v2_param = torch.load("/server19/msy/03-AD-perception/13-YOLOStereo3D/visualDet3D/depth_anything_v2_vitb.pth")
        # depth_v2_param从提出key含有pretrained的部分并组成新的dict
        pretrained_param = {k.replace("pretrained.", ""): v for k, v in depth_v2_param.items() if "pretrained." in k}
        depth_head_param = {k.replace("depth_head.", ""): v for k, v in depth_v2_param.items() if "depth_head." in k}
        self.pretrained.load_state_dict(pretrained_param)
        self.depth_head.load_state_dict(depth_head_param)

        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        # x = self.backbone(x['image'])
        x = x['image']
        ori_h, ori_w = input_depth.shape[-2] // 14, input_depth.shape[-1] // 14
        feature_resnet = self.backbone(x)
        input_depth = F.interpolate(x, (280, 1036), mode="bilinear", align_corners=False)
        with torch.no_grad():
            patch_h, patch_w = input_depth.shape[-2] // 14, input_depth.shape[-1] // 14
            features = self.pretrained.get_intermediate_layers(input_depth, self.intermediate_layer_idx[self.encoder], return_class_token=True)
            depth, depth_feature = self.depth_head(features, patch_h, patch_w)
        depth_feature = self.output_conv2(depth_feature)
    
        # save depth to imga.jpg
        # torchvision.utils.save_image(depth, "imgage_depth.jpg", normalize=True, scale_each=True)
        # torchvision.utils.save_image(x, "image_rgb.jpg", normalize=True, scale_each=True)
        # torchvision.utils.save_image(input_depth, "image_depth_input.jpg", normalize=True, scale_each=True)
        # return 0
        # cat feature_resnet[0] and layer_3_resnet to x
        # print("laylayer_3_resneter_3", layer_3_resnet.shape)
        # print("feature_resnet[0]", feature_resnet[0].shape)
        x = torch.cat([feature_resnet[0], depth_feature], dim=1)
        # x = feature_resnet[0]
        # x = depth_feature
        return x
