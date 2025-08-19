from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()
cfg.eval_only = False
# train data
cfg.obj_types_AR = ['barrel', 'sofa_bed', 'armoire', 'toaster_oven', 
                    'volleyball', 'air_conditioner', 'wheel', 'shoulder_bag', 
                    'cabinet', 'file_cabinet', 'bed', 'wardrobe', 'oven', 
                    'backpack', 'automatic_washer', 'reamer_(juicer)', 
                    'duffel_bag', 'printer', 'fire_extinguisher', 'projector', 
                    'wine_bucket', 'soccer_ball', 'bucket', 'water_heater', 
                    'bathtub', 'suitcase', 'refrigerator', 'cube', 'microwave_oven', 
                    'baby_buggy', 'sofa', 'television_set', 'satchel', 'wagon_wheel', 
                    'water_cooler', 'record_player', 'table', 'ice_maker', 'bookcase'] # will be mapped to "OtherForeground"
cfg.obj_types_kitti = ['Car', 'Cyclist', "Pedestrian"]    # train for classification loss
cfg.obj_types = cfg.obj_types_kitti # train for 3d+2d loss
cfg.obj_types_fg = cfg.obj_types + ["OtherForeground", "Ood"] # for 2d fg loss only


# test data
cfg.obj_types_test_Ood = ['parking_meter', 'polar_bear', 'horse', 'bulldog', 'mailbox_(at_home)', 
                          'camel', 'hippopotamus', 'postbox_(public)', 'pony', 'lamb_(animal)', 
                          'milestone', 'hog', 'telephone_booth', 'flowerpot', 'ram_(animal)', 
                          'sculpture', 'puppy', 'bench', 'shepherd_dog', 'wolf', 'giant_panda', 
                          'snowman', 'bear', 'giraffe', 'trunk', 'sheep', 'zebra', 'deer', 
                          'goat', 'manatee', 'horned_cow', 'gravestone', 'monkey', 'dumpster', 
                          'fireplug', 'gorilla', 'dalmatian', 'vending_machine', 'wheelchair', 
                          'tiger', 'chaise_longue', 'mail_slot', 'cow', 'dog', 'baboon', 
                          'rhinoceros', 'grizzly', 'trash_can', 'mammoth', 'cougar', 'person', 
                          'chair', 'pug-dog', 'scarecrow', 'alligator', 'lion', 'elephant', 
                          'statue_(sculpture)']  # will be mapped to "Ood" for test

cfg.obj_types_test_Ood += ['OtherForeground'] # will be mapped to "Ood"
cfg.obj_types_test = ['Car', 'Cyclist',  "Pedestrian", 'Ood'] # for test result

cfg.anchor_prior = True
## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 30, # for validation epoch 50 is enough
    disp_iter = 10,
    save_iter = 10,
    test_iter = 10,
    training_func = "train_stereo_detection",
    test_func = "test_stereo_detection",
    evaluate_func = "evaluate_kitti_obj",
)

cfg.trainer = trainer

## path
path = edict()
# server_prefix = "/server19/mushiyi/"
# server_prefix = "/server9/msy/"
# server_prefix = "/data1/mushiyi/" # server23
server_prefix = "/data3/mushiyi/" # server27
server_prefix_final = "/server19/mushiyi/" # server19
# server_prefix = "/data2"
# path.data_path = "smb9_msy/03-Data/02-KITTI/object/training/" # ori
path.data_path = "smb9_msy/03-Data/02-KITTI/KITTI-OOD/id-train"
# path.data_path = "smb9_msy/03-Data/02-KITTI/object/kitti_stereo_AR_train/Drop_item_train_and_ori" # indoor 3D AR

path.data_path = os.path.join(server_prefix_final, path.data_path)


# path.val_path = "smb9_msy/03-Data/02-KITTI/object/training/"
path.val_path = "smb9_msy/03-Data/02-KITTI/KITTI-OOD/ood-val"
path.val_path = os.path.join(server_prefix_final, path.val_path)

path.test_path = "smb9_msy/03-Data/02-KITTI/object/training/" # used in visualDet3D/data/.../dataset
path.test_path = os.path.join(server_prefix_final, path.test_path)
# path.data_path = "/dataset/067-msy/001-KITTI/object/training" # used in visualDet3D/data/.../dataset
# path.test_path = "/dataset/067-msy/001-KITTI/object/testing" # used in visualDet3D/data/.../dataset
path.workspace = "smb9_msy/02-Code/04-AD4AD/02-Ood3D/"
path.workspace = os.path.join(server_prefix_final, path.workspace)
path.visualDet3D_path =  os.path.join(path.workspace, "visualDet3D") # The path should point to the inner subfolder
path.project_path = os.path.join(path.workspace, "workdirs" )
if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)
path.project_path = os.path.join(path.project_path, '07-A3D-for-paper-r1', "301-ar-train-arood-val-2d-only")


if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)

path.log_path = os.path.join(path.project_path, "log")
if not os.path.isdir(path.log_path):
    os.mkdir(path.log_path)

path.checkpoint_path = os.path.join(path.project_path, "checkpoint")
if not os.path.isdir(path.checkpoint_path):
    os.mkdir(path.checkpoint_path)

path.preprocessed_path = os.path.join(path.project_path, "output")
if not os.path.isdir(path.preprocessed_path):
    os.mkdir(path.preprocessed_path)

path.train_imdb_path = os.path.join(path.preprocessed_path, "training")
if not os.path.isdir(path.train_imdb_path):
    os.mkdir(path.train_imdb_path)

path.val_imdb_path = os.path.join(path.preprocessed_path, "validation")
if not os.path.isdir(path.val_imdb_path):
    os.mkdir(path.val_imdb_path)

cfg.path = path

## optimizer
optimizer = edict(
    type_name = 'adam',
    keywords = edict(
        lr        = 1e-4,
        weight_decay = 0,
    ),
    clipped_gradient_norm = 0.1
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    type_name = 'CosineAnnealingLR',
    keywords = edict(
        T_max     = cfg.trainer.max_epochs,
        eta_min   = 5e-6,
    )
)
cfg.scheduler = scheduler

## data
# train_split_file = "smb9_msy/03-Data/02-KITTI/object/training/ImageSets/train.txt" # ori kitti
train_split_file = "smb9_msy/03-Data/02-KITTI/KITTI-OOD/id-train/ImageSets/train.txt" 
# train_split_file = "smb9_msy/03-Data/02-KITTI/object/kitti_stereo_AR_train/Drop_item_train_and_ori/ImageSets/train.txt" #indoor3d
train_split_file = os.path.join(server_prefix_final, train_split_file)
val_split_file = "smb9_msy/03-Data/02-KITTI/KITTI-OOD/ood-val/ImageSets/ood_val.txt"
# val_split_file = "smb9_msy/03-Data/02-KITTI/object/training/ImageSets/val.txt"
val_split_file = os.path.join(server_prefix_final, val_split_file)
data = edict(
    batch_size = 8,
    num_workers = 8,
    rgb_shape = (288, 1280, 3),
    train_dataset = "KittiStereoDataset",
    val_dataset   = "KittiStereoDataset",
    test_dataset  = "KittiStereoTestDataset",
    train_split_file = train_split_file,
    val_split_file   = val_split_file,
    replace_dir = [server_prefix, server_prefix_final],
)

data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    crop_top = 100,
    mix_up   = 0,
    get_disparity_on_the_fly = False
)
data.train_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='PhotometricDistort', keywords=edict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data = data

## networks
detector = edict()
detector.obj_types = cfg.obj_types
detector.name = 'Stereo3D'
detector.backbone = edict(
    depth=34,
    pretrained=True,
    frozen_stages=-1,
    num_stages=3,
    out_indices=(0, 1, 2),
    norm_eval=True,
    dilations=(1, 1, 1),
)
head_loss = edict(
    fg_iou_threshold = 0.5,
    bg_iou_threshold = 0.4,
    L1_regression_alpha = 5 ** 2,
    focal_loss_gamma = 2.0,
    balance_weight = [20.0, 40.0, 40.0], # 长度等于obj_types_kitti长度
    # regression_weight = [1, 1, 1, 1, 1, 1, 12, 1, 1, 0.5, 0.5, 0.5, 1], #[x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
    regression_weight = [1, 1, 1, 1, 1, 1, 12, 1, 1, 0.5, 0.5, 0.5, 1], #[x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
)
head_test = edict(
    score_thr=0.5,
    cls_agnostic = False,
    nms_iou_thr=0.4,
    post_optimization=False
)

anchors = edict(
        {
            'obj_types': cfg.obj_types_fg,
            'pyramid_levels':[4],
            'strides': [2 ** 4],
            'sizes' : [24],
            'ratios': np.array([0.5, 1, 2.0]),
            'scales': np.array([2 ** (i / 4.0) for i in range(16)]),
        }
    )

head_layer = edict(
    num_features_in=1408,
    num_cls_output=len(cfg.obj_types_fg)+1,
    num_reg_output=12,
    cls_feature_size=256,
    reg_feature_size=1408,
    fg_feature_size=128,
)
detector.head = edict(
    num_regression_loss_terms=13,
    preprocessed_path=path.preprocessed_path,
    num_classes     = len(cfg.obj_types_fg),
    anchors_cfg     = anchors,
    layer_cfg       = head_layer,
    loss_cfg        = head_loss,
    test_cfg        = head_test,
    read_precompute_anchor = True,
    ood_score_type = "RBAF" # ["MSPF", "RBAF"]

)
detector.anchors = anchors
detector.loss = head_loss
cfg.detector = detector
