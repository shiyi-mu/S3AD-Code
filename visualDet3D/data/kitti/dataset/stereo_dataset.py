from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import skimage.measure
import random
import csv
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
from visualDet3D.data.kitti.kittidata import KittiData, KittiObj, KittiCalib
from visualDet3D.data.pipeline import build_augmentator

import os
import pickle
import numpy as np
from copy import deepcopy
from visualDet3D.utils.utils import alpha2theta_3d, theta2alpha_3d, draw_3D_box
from visualDet3D.networks.utils import BBox3dProjector
from visualDet3D.networks.utils.registry import DATASET_DICT
import sys
from matplotlib import pyplot as plt
from PIL import Image

ros_py_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if sys.version_info > (3, 0) and ros_py_path in sys.path:
    #Python 3, compatible with a naive ros environment
    sys.path.remove(ros_py_path)
    import cv2
    sys.path.append(ros_py_path)
else:
    #Python 2
    import cv2

@DATASET_DICT.register_module
class KittiStereoDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, cfg, split='training', imdb_file_path=None):
        super(KittiStereoDataset, self).__init__()
        preprocessed_path   = cfg.path.preprocessed_path
        obj_types           = cfg.obj_types
        obj_types_fg        = cfg.obj_types_fg
        aug_cfg             = cfg.data.augmentation
        is_train = (split == 'training')
        if imdb_file_path is None:
            imdb_file_path = os.path.join(preprocessed_path, split, 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
                "calib": True,
                "image": True,
                "image_3":True,
                "label": False,
                "label3": False,
                "velodyne": False
            }
        if is_train:
            self.transform = build_augmentator(cfg.data.train_augmentation)
            print("debug >>> augcfg", aug_cfg)
            self.random_mixup3d = aug_cfg["mix_up"]
            self.get_disparity_on_the_fly = aug_cfg["get_disparity_on_the_fly"]
        else:
            self.transform = build_augmentator(cfg.data.test_augmentation)
            self.random_mixup3d = 0
            self.get_disparity_on_the_fly = False
        self.projector = BBox3dProjector()
        self.is_train = is_train
        self.obj_types = obj_types
        self.obj_types_fg = obj_types_fg
        self.preprocessed_path = preprocessed_path
        self.stds = cfg.data.augmentation.rgb_std
        self.means = cfg.data.augmentation.rgb_mean
        if cfg.data.replace_dir is not None:
            self.replace_dir = cfg.data.replace_dir

    def _reproject(self, P2:np.ndarray, transformed_label:List[KittiObj]) -> Tuple[List[KittiObj], np.ndarray]:
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label) > 0:
            #for obj in transformed_label:
            #    obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)
            bbox3d_origin = torch.tensor([[obj.x, obj.y - 0.5 * obj.h, obj.z, obj.w, obj.h, obj.l, obj.alpha] for obj in transformed_label], dtype=torch.float32)
            try:
                abs_corner, homo_corner, _ = self.projector.forward(bbox3d_origin, bbox3d_origin.new(P2))
            except:
                print('\n',bbox3d_origin.shape, len(transformed_label), transformed_label, bbox3d_origin)
            for i, obj in enumerate(transformed_label):
                extended_center = np.array([obj.x, obj.y - 0.5 * obj.h, obj.z, 1])[:, np.newaxis] #[4, 1]
                extended_bottom = np.array([obj.x, obj.y, obj.z, 1])[:, np.newaxis] #[4, 1]
                image_center = (P2 @ extended_center)[:, 0] #[3]
                image_center[0:2] /= image_center[2]

                image_bottom = (P2 @ extended_bottom)[:, 0] #[3]
                image_bottom[0:2] /= image_bottom[2]
                
                bbox3d_state[i] = np.concatenate([image_center,
                                                 [obj.w, obj.h, obj.l, obj.alpha]]) #[7]

            max_xy, _= homo_corner[:, :, 0:2].max(dim = 1)  # [N,2]
            min_xy, _= homo_corner[:, :, 0:2].min(dim = 1)  # [N,2]

            result = torch.cat([min_xy, max_xy], dim=-1) #[:, 4]

            bbox2d = result.cpu().numpy()

            for i in range(len(transformed_label)):
                # print("0  ori transformed_label[i].bbox_l", transformed_label[i].bbox_l)
                transformed_label[i].bbox_l = bbox2d[i, 0]
                # print("0after transformed_label[i].bbox_l", transformed_label[i].bbox_l)
                # print("1  ori transformed_label[i].bbox_t", transformed_label[i].bbox_t)
                transformed_label[i].bbox_t = bbox2d[i, 1]
                # print("1after transformed_label[i].bbox_t", transformed_label[i].bbox_t)
                # print("2  ori transformed_label[i].bbox_r", transformed_label[i].bbox_r)
                transformed_label[i].bbox_r = bbox2d[i, 2]
                # print("2after transformed_label[i].bbox_r", transformed_label[i].bbox_r)
                # print("3  ori transformed_label[i].bbox_b", transformed_label[i].bbox_b)
                transformed_label[i].bbox_b = bbox2d[i, 3]
                # print("3after transformed_label[i].bbox_b", transformed_label[i].bbox_b)
        return transformed_label, bbox3d_state
    
    def denormalize(self, image):
        image_np =  np.array(image)
        image_np = ((image_np * self.stds + self.means)*255).astype(np.uint8)
        return Image.fromarray(image_np)
    
    def get_disparty_P2_onfly(self, img_left, img_right, flip_flag):
        left_image = cv2.cvtColor(np.asarray(img_left),cv2.COLOR_RGB2BGR)
        right_image = cv2.cvtColor(np.asarray(img_right),cv2.COLOR_RGB2BGR)
        gray_image1 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        matcher = cv2.StereoBM_create(192, 25)
        # if flip_flag:
        #     gray_image1_flip = cv2.flip(gray_image1, 1)
        #     gray_image2_flip = cv2.flip(gray_image2, 1)
        #     disparity_left = matcher.compute(gray_image1_flip, gray_image2_flip)
        #     disparity_left = disparity_left[:,::-1]
        #     disparity_left[disparity_left < 0] = 0
        # else :
        disparity_left = matcher.compute(gray_image1, gray_image2)
        disparity_left[disparity_left < 0] = 0
            
        disparity_left = disparity_left.astype(np.uint16)
        disparity_left = skimage.measure.block_reduce(disparity_left, (4,4), np.max) 
        # disparity_left = Image.fromarray(disparity_left) 
        return disparity_left
    
    def expand_bbox(self, label_dino):
        for obj in label_dino:
            center_x = (obj.bbox_l + obj.bbox_r)/2.0
            center_y = (obj.bbox_t + obj.bbox_b)/2.0
            new_w =  (obj.bbox_r - obj.bbox_l)*1.4
            new_h =  (obj.bbox_b - obj.bbox_t)*1.0
            obj.bbox_l = center_x - new_w*0.5
            obj.bbox_r = center_x + new_w*0.5
            obj.bbox_t = center_y - new_h*0.5
            obj.bbox_b = center_y + new_h*0.5

    def __getitem__(self, index):
        kitti_data = self.imdb[index]
        # The calib and label has been preloaded to minimize the time in each indexing
        kitti_data.output_dict = self.output_dict
        if self.replace_dir is not None:
            kitti_data.replace_dir_prefix(self.replace_dir[0], self.replace_dir[1])   
        # print(">>>>>>>"*4)
        calib, left_image, right_image, _label, _, _label3 = kitti_data.read_data()
        # print("_label", _label)
        # print("_label3", _label3)
        calib.image_shape = left_image.shape
        label = []
        random_mix_flag = False
        if np.random.random() < self.random_mixup3d:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = int(np.random.choice(range(0,len(self.imdb))))
                kitti_data_temp = self.imdb[random_index]
                kitti_data_temp.output_dict = self.output_dict
                calib_temp, left_image_temp, right_image_temp, _, _, _ = kitti_data_temp.read_data()
  
                if calib_temp.P2[0, 2] == calib.P2[0, 2] \
                    and calib_temp.P2[1, 2] == calib.P2[1, 2] \
                        and calib_temp.P2[0, 0] == calib.P2[0, 0] \
                            and calib_temp.P2[1, 1] == calib.P2[1, 1]:

                    if  left_image_temp.shape[0] == left_image.shape[0] and left_image_temp.shape[1] == left_image.shape[1]:
                        objects_1 = kitti_data.label
                        objects_2 = kitti_data_temp.label
                        if len(objects_1) + len(objects_2) < 50: 
                            random_mix_flag = True
                            alpha = 0.5
                            left_image = (1 - alpha) * left_image + alpha * left_image_temp
                            right_image = (1 - alpha) * right_image + alpha * right_image_temp
                            break
        for obj in kitti_data.label:
            # if obj.type in self.obj_types:
            if obj.type in self.obj_types_fg:
                label.append(obj)

        label_dino = []
        for obj in kitti_data.label:
            # if obj.type in self.obj_types:
            if (obj.type in self.obj_types_fg) and (obj.type not in self.obj_types):
                label_dino.append(obj)
        
        # print("label type", [obj.type  for obj in label])
        # print("label bbox_l", [[obj.bbox_l,obj.bbox_t,obj.bbox_r,obj.bbox_b]   for obj in label])
        label3 = []
        if kitti_data.label3 is not None:
            for obj in kitti_data.label3:
                # if obj.type in self.obj_types:
                if obj.type in self.obj_types_fg:
                    label3.append(obj)

        if random_mix_flag:
            for obj in kitti_data_temp.label:
                if obj.type in self.obj_types_fg:
                    label.append(obj)
                if (obj.type in self.obj_types_fg) and (obj.type not in self.obj_types):
                    label_dino.append(obj)
            if kitti_data_temp.label3 is not None:
                for obj in kitti_data_temp.label3:
                    if obj.type in self.obj_types_fg:
                        label3.append(obj)

        transformed_left_image, transformed_right_image, P2, P3, transformed_label_all , transformed_label_3= self.transform(
                left_image, right_image, deepcopy(calib.P2), deepcopy(calib.P3), deepcopy(label), deepcopy(label3)
        )
        if self.get_disparity_on_the_fly:
            transformed_left_image_denormalize = self.denormalize(transformed_left_image)
            transformed_right_image_denormalize = self.denormalize(transformed_right_image)
            if abs(P2[0, 3]) < abs(P3[0, 3]):
                flip_flag = False
            else:
                flip_flag = True
            disparity = self.get_disparty_P2_onfly(transformed_left_image_denormalize, transformed_right_image_denormalize, flip_flag)
        transformed_label = [obj  for obj in transformed_label_all if (obj.type in self.obj_types)]
        transformed_label_extand = [obj  for obj in transformed_label_all if (obj.type in self.obj_types_fg and obj.type not in self.obj_types)]
        self.expand_bbox(transformed_label_extand)

        transformed_label_3 =  [obj for obj in transformed_label_3 if (obj.type in self.obj_types_fg)]
        self.expand_bbox(transformed_label_3)
        bbox3d_state = np.zeros([len(transformed_label), 7]) #[camera_x, camera_y, z, w, h, l, alpha]
        if len(transformed_label) > 0:
            transformed_label, bbox3d_state = self._reproject(P2, transformed_label)
        # if len(transformed_label_extand) > 0:
        #     transformed_label_extand, _ = self._reproject(P2, transformed_label_extand)
        # label_2ds = transformed_label + transformed_label_extand
        if self.is_train:
            if abs(P2[0, 3]) < abs(P3[0, 3]): # not mirrored or swaped, disparity should base on pointclouds projecting through P2
                # disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P2%06d.png" % index), -1)
                if self.get_disparity_on_the_fly:
                    pass
                else:
                    disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P2{}.png".format(kitti_data.index_name)), -1)
                label_2ds = transformed_label + transformed_label_extand
            else: # mirrored and swap, disparity should base on pointclouds projecting through P3, and also mirrored
                # disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P3%06d.png" % index), -1)
                if self.get_disparity_on_the_fly:
                    pass
                else:
                    disparity = cv2.imread(os.path.join(self.preprocessed_path, 'training', 'disp', "P3{}.png".format(kitti_data.index_name)), -1)
                label_2ds = transformed_label + transformed_label_3
                # disparity = disparity[:, ::-1]
            disparity = disparity / 16.0
        else:
            disparity = None
            label_2ds = transformed_label + transformed_label_extand

        bbox2d = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in transformed_label])
        bbox2d_ext = np.array([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label_2ds])
        output_dict = {'calib': [P2, P3],
                       'image': [transformed_left_image, transformed_right_image],
                       'label': [obj.type for obj in transformed_label], 
                       'label_ext': [obj.type for obj in label_2ds], 
                       'bbox2d': bbox2d, #[N, 4] [x1, y1, x2, y2]
                       'bbox2d_ext': bbox2d_ext,
                       'bbox3d': bbox3d_state,
                       'original_shape': calib.image_shape,
                       'disparity': disparity,
                       'original_P':calib.P2.copy(),
                       'index_name': kitti_data.index_name}
        # print("output_dict[index_name]", output_dict["index_name"])
        # print("output_dict[label]", output_dict["label"])
        # print("output_dict[label_ext]", output_dict["label_ext"])
        # print("output_dict[bbox2d]", output_dict["bbox2d"])
        # print("output_dict[bbox2d_ext]", output_dict["bbox2d_ext"])
        return output_dict

    def __len__(self):
        return len(self.imdb)

    @staticmethod
    def collate_fn(batch):
        left_images = np.array([item["image"][0] for item in batch])#[batch, H, W, 3]
        left_images = left_images.transpose([0, 3, 1, 2])

        right_images = np.array([item["image"][1] for item in batch])#[batch, H, W, 3]
        right_images = right_images.transpose([0, 3, 1, 2])

        P2 = np.array([item['calib'][0] for item in batch])
        P3 = np.array([item['calib'][1] for item in batch])
        label = [item['label'] for item in batch]
        label_ext = [item['label_ext'] for item in batch]
        bbox2ds = [item['bbox2d'] for item in batch]
        bbox2ds_ext = [item['bbox2d_ext'] for item in batch]
        bbox3ds = [item['bbox3d'] for item in batch]
        disparities = np.array([item['disparity'] for item in batch])
        if disparities[0] is None:
            return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float(), torch.tensor(P2).float(), torch.tensor(P3).float(), label, bbox2ds, bbox3ds, label_ext, bbox2ds_ext
        else:
            return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float(), torch.tensor(P2).float(), torch.tensor(P3).float(), label, bbox2ds, bbox3ds, torch.tensor(disparities).float(), label_ext, bbox2ds_ext

@DATASET_DICT.register_module
class KittiStereoTestDataset(KittiStereoDataset):
    def __init__(self, cfg, split='test'):
        preprocessed_path   = cfg.path.preprocessed_path
        obj_types           = cfg.obj_types
        aug_cfg             = cfg.data.augmentation
        super(KittiStereoTestDataset, self).__init__(cfg, split)
        imdb_file_path = os.path.join(preprocessed_path, 'test', 'imdb.pkl')
        self.imdb = pickle.load(open(imdb_file_path, 'rb')) # list of kittiData
        self.output_dict = {
                "calib": True,
                "image": True,
                "image_3":True,
                "label": False,
                "velodyne": False
            }

    def __getitem__(self, index):
        kitti_data = self.imdb[index]
        # The calib and label has been preloaded to minimize the time in each indexing
        kitti_data.output_dict = self.output_dict
        calib, left_image, right_image, _, _ = kitti_data.read_data()
        calib.image_shape = left_image.shape

        transformed_left_image, transformed_right_image, P2, P3 = self.transform(
                left_image, right_image, deepcopy(calib.P2),deepcopy(calib.P3)
        )

        output_dict = {'calib': [P2, P3],
                       'image': [transformed_left_image, transformed_right_image],
                       'original_shape': calib.image_shape,
                       'original_P':calib.P2.copy()}
        return output_dict

    @staticmethod
    def collate_fn(batch):
        left_images = np.array([item["image"][0] for item in batch])#[batch, H, W, 3]
        left_images = left_images.transpose([0, 3, 1, 2])

        right_images = np.array([item["image"][1] for item in batch])#[batch, H, W, 3]
        right_images = right_images.transpose([0, 3, 1, 2])

        P2 = [item['calib'][0] for item in batch]
        P3 = [item['calib'][1] for item in batch]
        return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float(), P2, P3

