import time
import fire
import visualDet3D.evaluator.kitti.kitti_common as kitti
from .eval import get_coco_eval_result, get_official_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_classes=0,
             Ood_classes=[],
             coco=False,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    # map new class2ood
    print("Ood_classes", Ood_classes)
    for gt_anno in gt_annos:
        for i, name in enumerate(gt_anno['name']):
           if name in Ood_classes:
               gt_anno['name'][i] = 'Ood'
    for dt_anno in dt_annos:
        for i, name in enumerate(dt_anno['name']):
           if name in Ood_classes:
               dt_anno['name'][i] = 'Ood'
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_classes)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_classes)


if __name__ == '__main__':
    fire.Fire()
