"""
Find thresholds (used to binarize the heatmaps) that maximize mIoU on the
validation set. Pass in a list of potential thresholds [0.2, 0.3, ... , 0.8].
Save the best threshold for each pathology in a csv file.
"""
# based on https://github.com/rajpurkarlab/cheXlocalize/blob/v1.0.0/tune_heatmap_threshold.py
from argparse import ArgumentParser
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from pycocotools import mask
import torch.nn.functional as F
from tqdm import tqdm

from eval import calculate_iou
from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask
from functools import lru_cache #added to code
import multiprocessing as mp


###added for speedup
def computeOneIou(inp):
    threshold,pkl_path,gt = inp
    path = str(pkl_path).split('/')
    task = path[-1].split('_')[-2]
    img_id = '_'.join(path[-1].split('_')[:-2])

    # add image and segmentation to submission dictionary
    if img_id in gt:
        pred_mask = pkl_to_mask(pkl_path=pkl_path, threshold=threshold)
        gt_item = gt[img_id][task]
        gt_mask = mask.decode(gt_item)
        assert (pred_mask.shape == gt_mask.shape)
        true_pos_only = args.true_pos_only == "True"
        iou_score = calculate_iou(pred_mask, gt_mask, true_pos_only=true_pos_only)
    else:
        iou_score = np.nan
   

    return iou_score

def compute_miou_parallel(threshold, cam_pkls, gt):
    """
    Given a threshold and a list of heatmap pickle files, return the mIoU.

    Args:
        threshold (double): the threshold used to convert heatmaps to segmentations
        cam_pkls (list): a list of heatmap pickle files (for a given pathology)
        gt (dict): dictionary of ground truth segmentation masks
    """

    inpList = [(threshold,pkl_path,gt) for pkl_path in cam_pkls]

    with mp.Pool() as pool:
        ious = pool.map(computeOneIou,inpList)
    print(len(ious),len([x for x in ious if not np.isnan(x)]),len(gt.keys()))
    miou = np.nanmean(np.array(ious))

    return miou


def compute_miou(threshold, cam_pkls, gt):
    """
    Given a threshold and a list of heatmap pickle files, return the mIoU.

    Args:
        threshold (double): the threshold used to convert heatmaps to segmentations
        cam_pkls (list): a list of heatmap pickle files (for a given pathology)
        gt (dict): dictionary of ground truth segmentation masks
    """
    ious = []
    for pkl_path in tqdm(cam_pkls):
        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])

        # add image and segmentation to submission dictionary
        if img_id in gt:
            pred_mask = pkl_to_mask(pkl_path=pkl_path, threshold=threshold)
            gt_item = gt[img_id][task]
            gt_mask = mask.decode(gt_item)
            assert (pred_mask.shape == gt_mask.shape)
            iou_score = calculate_iou(pred_mask, gt_mask, true_pos_only=args.true_pos_only)
        else:
            iou_score = np.nan
        ious.append(iou_score)
        
    miou = np.nanmean(np.array(ious))
    return miou

def tune_threshold(task, gt, cam_dir):
    """
    For a given pathology, find the threshold that maximizes mIoU.

    Args:
        task (str): localization task
        gt (dict): dictionary of the ground truth segmentation masks
        cam_dir (str): directory with pickle files containing heat maps
    """
    cam_pkls = sorted(list(Path(cam_dir).rglob(f"*{task}_map.pkl")))

    #old code: (uses linear search)
    # thresholds = np.arange(0.2, .8, .1)
    # mious = [compute_miou(threshold, cam_pkls, gt) for threshold in thresholds]
    # best_threshold = thresholds[mious.index(max(mious))]

    #added code uses ternary search
    @lru_cache
    def miouForThreshold(threshold):
        return compute_miou_parallel(threshold, cam_pkls, gt)
    
    stepSize = 0.25
    best_threshold = 0.5
    roundNum = 10
    for round in range(roundNum):
        thresholdList = [best_threshold+(x*stepSize) for x in [-1,0,1]]
        mious = [(miouForThreshold(x),x) for x in thresholdList]
        mious = [x for x in mious if not np.isnan(x[0])]
        if len(mious) == 0: break
        print(mious)
        foundPair = sorted(mious,key=lambda x: x[0])[-1]
        best_threshold = foundPair[1]
        print("round",round,"optThreshold",foundPair)
        stepSize = stepSize/2
    ###added code ends here###
        


    return best_threshold


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--map_dir', type=str,
                        help='directory with pickle files containing heat maps')
    parser.add_argument('--gt_path', type=str,
                        help='json file where ground-truth segmentations are \
                              saved (encoded)')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='where to save the best thresholds tuned on the \
                              validation set')
    parser.add_argument('--true_pos_only', type=str, default='True',
                        help='if true, run evaluation only on the true positive \
                        slice of the dataset (CXRs that contain predicted and \
                        ground-truth segmentations); if false, also include cxrs \
                        with a predicted segmentation but without a ground-truth \
                        segmentation, and include cxrs with a ground-truth\
                        segmentation but without a predicted segmentation.')
    args = parser.parse_args()

    with open(args.gt_path) as f:
        gt = json.load(f)

    # tune thresholds and save the best threshold for each pathology to a csv file
    tuning_results = pd.DataFrame(columns=['threshold', 'task'])
    for task in sorted(LOCALIZATION_TASKS):
        print(f"Task: {task}")
        threshold = tune_threshold(task, gt, args.map_dir)
        #old code: (rounds threshold search)
        # df = pd.DataFrame([[round(threshold, 1), task]],
        #                   columns=['threshold', 'task'])
        #old code end
        df = pd.DataFrame([[threshold, task]],
                            columns=['threshold', 'task'])
        tuning_results = pd.concat([tuning_results, df], axis=0)

    tuning_results.to_csv(f'{args.save_dir}/tuning_results.csv', index=False)
