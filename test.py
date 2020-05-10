import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

from model import DispNet_sequential as DispNet

from params import *

from pyramid import scale_pyramid

class KITTIDataset(Dataset):
    def __init__(self):
        self.rootdir = test_dataset_location
        self.left_images = []
        self.ground_truth_disparity_images = []
        self.height = 512
        self.length = 256
        self.total_left_images = 0
        self.total_ground_truth_disparity_images = 0
        for subdir, dirs, files in os.walk(self.rootdir):
            if "image_2" in subdir: #Left RGB Folder
                for file in files:
                    if "_10.png" in file:
                        left_file = os.path.join(subdir, file)
                        self.left_images.append(left_file)
                        self.total_left_images += 1
                
            if "disp_noc_0" in subdir: # GT disparity folder
                for file in files:
                    if ".png" in file:
                        gt_file = os.path.join(subdir, file)
                        self.ground_truth_disparity_images.append(gt_file)
                        self.total_ground_truth_disparity_images += 1
        
        self.left_images.sort()
        self.ground_truth_disparity_images.sort()
        assert len(self.left_images) == len(self.ground_truth_disparity_images)
        print("Total Stereo images acquired: ", self.total_left_images)       

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = cv2.imread(self.left_images[idx])
        ground_truth_disp = cv2.imread(self.ground_truth_disparity_images[idx],cv2.IMREAD_GRAYSCALE)

        np.asarray(left_img)
        np.asarray(ground_truth_disp)
        
        left_img = cv2.resize(left_img,(self.height, self.length))

        left_img = np.moveaxis(left_img, 2,0)

        return {"left_img":left_img,"ground_truth_disp":ground_truth_disp}

"""
For Result Comparison
Return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
"""
def compute_errors(ground_truth_depth, predicted_depth):
    thresh = np.maximum((ground_truth_depth / predicted_depth), (predicted_depth / ground_truth_depth))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (ground_truth_depth - predicted_depth) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(ground_truth_depth) - np.log(predicted_depth)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(ground_truth_depth - predicted_depth) / ground_truth_depth)

    sq_rel = np.mean(((ground_truth_depth - predicted_depth)**2) / ground_truth_depth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

"""
Using fb/Disp -> generate Depth
"""
def convert_disps_to_depths_kitti(ground_truth_disparities, predicted_disparities):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []
    
    for i in range(len(ground_truth_disparities)):
        gt_disp = ground_truth_disparities[i]
        height, width = gt_disp.shape

        pred_disp = predicted_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
        pred_disparities_resized.append(pred_disp) 

        gt_mask = gt_disp > 0
        pred_mask = pred_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - gt_mask))
        pred_depth = width_to_focal[width] * 0.54 / (pred_disp + (1.0 - pred_mask))

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized


if __name__ == '__main__':
    print("\n \n --- Monocular Depth estimation train code --- \n \n")
    
    # load dataset 
    dataset = KITTIDataset()
    TestLoader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = MANY_WORKERS)
    # load model
    net = DispNet()
    net.to(torch.device(compute_device))
    net.load_state_dict(torch.load(pth_file_location))

    gt_disparities = []
    pred_disparities = []
    for batch, sample_batched in enumerate(TestLoader):
        with torch.no_grad():    
            left_original = sample_batched["left_img"]
            gt_original = sample_batched["ground_truth_disp"]

            if is_gpu_available: 
                left = left_original.type(torch.FloatTensor).cuda()
            else:
                left = left_original.type(torch.FloatTensor)

            left_pyramid = scale_pyramid(left,4)

            # Forward pass
            output = net.forward(left)

            left_disp = [output[i][:, 0, :, :] for i in range(4)]
            right_disp = [output[i][:, 1, :, :] for i in range(4)]
            rgb = right_disp[0][0].detach().cpu().numpy()
            rgb = cv2.resize(rgb,(gt_original[0].shape[1], gt_original[0].shape[0]))
            final = left_original[0].detach().cpu().numpy()

            """
            CHECK THIS RESULT
            """
            # gt_disparities.append(final.expand(3, gt_original[0].shape[1], gt_original[0].shape[0]))
            gt_disparities.append(final)
            pred_disparities.append(rgb)

            disparity_with_image = np.vstack((final,rgb))
            cv2.imwrite(save_test_images_in + str(batch) + ".jpg", disparity_with_image)

            """
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # fig.suptitle('Monodepth')
            # ax1.imshow(rgb,cmap='plasma')
            # ax2.imshow(final)
            # plt.imshow(rgb,cmap='plasma')
            # # plt.savefig('/home/bala/DeepLearning/data_scene_flow/training/Test_result1'+str(i+1)) 
            # plt.show()
            # plt.pause(0.5)
            """

    gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)
    num_samples = len(gt_disparities)
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        pred_disp = pred_disparities_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

    # Write-Overwrites 
    file1 = open(save_statistics_in,"w")#write mode 
    file1.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3')) 
    file1.write("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
    file1.close() 