# system imports
import os
import time
import sys 
sys.dont_write_bytecode = True #stop generating .pyc files
if not sys.warnoptions: #stop generating warnings
    import warnings
    warnings.simplefilter("ignore")

import datetime

# common matrix manipulation
import numpy as np

# plotting, Image showing, Image string operations
import matplotlib.pyplot as plt

# import Image
from PIL import Image

# Image loading from disk
import cv2

# Progress bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# model 
from model.model import DispNet_sequential as DispNet

# loss 
from loss.loss import reconstruct_using_disparity 
from loss.disparity_smoothness_loss import disparity_smoothness, temporal_disparity_smoothness
from loss.ssim_loss import SSIM_loss

# Pyramid generation
from utils.pyramid import scale_pyramid

from dataset.dataset import KITTIDataset_from_folder_for_temporal_smoothness, KITTIDataset_from_txt_for_temporal_smoothness
from params import *

if __name__ == '__main__':
    print("\n \n --- Monocular Depth Estimation train code --- \n \n")
    
    # load dataset 
    if source_of_training == "txt":
        dataset = KITTIDataset_from_txt_for_temporal_smoothness()
    else:
        dataset = KITTIDataset_from_folder_for_temporal_smoothness()

    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = MANY_WORKERS)
    
    # load model
    net = DispNet()
    net.to(torch.device(compute_device))

    if resume_trining:
        print("\n Loading previous weights from ", pth_file_location)
        net.load_state_dict(torch.load(pth_file_location))
    
    # configure loss
    print("\n \nTraining with the following loss parmeters:")
    print("appearance_matching_loss_weight: ",appearance_matching_loss_weight)
    print("LR_loss_weight: ", LR_loss_weight)
    print("disparity_smoothness_loss_weight: ", disparity_smoothness_loss_weight)
    print("alpha_appearance_matching_loss: ", alpha_appearance_matching_loss)
    print("temporal_disparity_smoothness_loss_weight: ", temporal_disparity_smoothness_loss_weight)
    print("\n")

    if is_gpu_available:
        loss_function = nn.L1Loss().cuda()
    else:
        loss_function = nn.L1Loss()

    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    current_datetime = datetime.datetime.now()
    print("Training Started @ ", current_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    for epoch in range(1, EPOCH+1):
        for batch_data in tqdm(TrainLoader):
            # retrieve stereo images
            left_original = batch_data["left_img"]
            prev_left_original = batch_data["prev_left_img"]
            right_original = batch_data["right_img"]
            prev_right_original = batch_data["prev_right_img"]

            # send to CUDA device
            if is_gpu_available: 
                left = left_original.type(torch.FloatTensor).cuda()
                prev_left = prev_left_original.type(torch.FloatTensor).cuda()
                right = right_original.type(torch.FloatTensor).cuda()
                prev_right = prev_right_original.type(torch.FloatTensor).cuda()
            else:
                left = left_original.type(torch.FloatTensor)
                prev_left = prev_left_original.type(torch.FloatTensor)
                right = right_original.type(torch.FloatTensor)
                prev_right = prev_right_original.type(torch.FloatTensor)

            # generate pyramid
            left_pyramid = scale_pyramid(left,4)
            right_pyramid = scale_pyramid(right,4)
            prev_left_pyramid = scale_pyramid(prev_left,4)
            prev_right_pyramid = scale_pyramid(prev_right,4)
            # forward pass with left image
            output = net.forward(left)
            prev_output = net.forward(prev_left)

            # collect disparities from the model
            left_disp = [output[i][:, 0, :, :] for i in range(4)]
            right_disp = [output[i][:, 1, :, :] for i in range(4)]
            prev_left_disp = [prev_output[i][:, 0, :, :] for i in range(4)]
            prev_right_disp = [prev_output[i][:, 1, :, :] for i in range(4)]

            # reconsturct corresponding images using disparities
            right_reconstuct = [reconstruct_using_disparity(left_pyramid[i], right_disp[i]) for i in range(4)]
            left_reconstuct = [reconstruct_using_disparity(right_pyramid[i], left_disp[i]) for i in range(4)]
            
            """
            calculate L1 loss
            """
            # TODO: Put weighted loss for pyramid : error in smaller image should contribute more
            left_L1loss = [loss_function(left_pyramid[i], left_reconstuct[i]) for i in range(4)]
            right_L1loss = [loss_function(right_pyramid[i], right_reconstuct[i]) for i in range(4)]
            if is_gpu_available:
                total_L1_loss = torch.FloatTensor([0]).cuda()
                total_SSIM_loss = torch.FloatTensor([0]).cuda()
            else:
                total_L1_loss = torch.FloatTensor([0])
                total_SSIM_loss = torch.FloatTensor([0])

            for i in range(4): 
                total_L1_loss += (left_L1loss[i] + right_L1loss[i])
            total_L1_loss /= 4 

            """
            calculate SSIM loss
            """
            left_SSIM_loss = [torch.mean(SSIM_loss(left_pyramid[i], left_reconstuct[i])) for i in range(4)] #Reconstructed Image and Original Image 
            right_SSIM_loss = [torch.mean(SSIM_loss(right_pyramid[i], right_reconstuct[i])) for i in range(4)]
            for i in range(4): 
                total_SSIM_loss += (left_SSIM_loss[i] + right_SSIM_loss[i])
            total_SSIM_loss /= 4
            
            """
            Total apparance matching loss
            """
            appearance_matching_loss = (alpha_appearance_matching_loss * total_SSIM_loss) + (1- alpha_appearance_matching_loss)*total_L1_loss
            # print("appearance matching loss: ", appearance_matching_loss)

            # append axis of channel to treat disparities as images
            left_disp[0] = left_disp[0].view([-1, 1, 256, 512])
            left_disp[1] = left_disp[1].view([-1, 1, 128, 256])
            left_disp[2] = left_disp[2].view([-1, 1, 64, 128])
            left_disp[3] = left_disp[3].view([-1, 1, 32, 64])

            prev_left_disp[0] = prev_left_disp[0].view([-1, 1, 256, 512])
            prev_left_disp[1] = prev_left_disp[1].view([-1, 1, 128, 256])
            prev_left_disp[2] = prev_left_disp[2].view([-1, 1, 64, 128])
            prev_left_disp[3] = prev_left_disp[3].view([-1, 1, 32, 64])

            right_disp[0] = right_disp[0].view([-1, 1, 256, 512])
            right_disp[1] = right_disp[1].view([-1, 1, 128, 256])
            right_disp[2] = right_disp[2].view([-1, 1, 64, 128])
            right_disp[3] = right_disp[3].view([-1, 1, 32, 64])

            prev_right_disp[0] = prev_right_disp[0].view([-1, 1, 256, 512])
            prev_right_disp[1] = prev_right_disp[1].view([-1, 1, 128, 256])
            prev_right_disp[2] = prev_right_disp[2].view([-1, 1, 64, 128])
            prev_right_disp[3] = prev_right_disp[3].view([-1, 1, 32, 64])

            """
            Calculate L-R consistency loss
            """
            reconstruct_left = [reconstruct_using_disparity(right_disp[i], left_disp[i]) for i in range(4)]
            reconstruct_right = [reconstruct_using_disparity(left_disp[i], right_disp[i]) for i in range(4)]    
            LR_loss_left = [torch.mean(left_disp[i]-reconstruct_left[i]) for i in range(4)]
            LR_loss_right = [torch.mean(right_disp[i]-reconstruct_right[i]) for i in range(4)]
            if is_gpu_available:
                total_LR_loss = torch.FloatTensor([0]).cuda()
            else:
                total_LR_loss = torch.FloatTensor([0])
            for i in range(4): 
                total_LR_loss += LR_loss_left[i] + LR_loss_right[i] 
            total_LR_loss /= 4

            """
            Disparity smoothness loss
            """
            disparity_smoothnesss_loss_left = disparity_smoothness(left_pyramid,left_disp)
            disparity_smoothness_loss_right = disparity_smoothness(right_pyramid,right_disp)
            disparity_smoothness_loss = sum(disparity_smoothnesss_loss_left + disparity_smoothness_loss_right)
            
            """
            Temporal Disparity Smoothness loss
            """
            temporal_disparity_smoothness_loss_left = temporal_disparity_smoothness(prev_left_pyramid, left_pyramid, prev_left_disp, left_disp)
            temporal_disparity_smoothness_loss_right = temporal_disparity_smoothness(prev_right_pyramid, right_pyramid, prev_right_disp, right_disp)
            temporal_disparity_smoothness_loss = sum(temporal_disparity_smoothness_loss_left + temporal_disparity_smoothness_loss_right)

            loss = (appearance_matching_loss_weight * appearance_matching_loss+ \
                LR_loss_weight * total_LR_loss + \
                disparity_smoothness_loss_weight * disparity_smoothness_loss +\
                temporal_disparity_smoothness_loss_weight* temporal_disparity_smoothness_loss)/BATCH_SIZE  

            loss.backward()  
            optimizer.step()
            net.zero_grad()
            scheduler.step() 

        #TO DO: Query same image and see how it evolves over epochs
        print("Epoch : ", epoch, " Loss: ", loss)
        rgb = right_disp[0][0].detach().cpu().numpy()
        fig = plt.figure(1)
        plt.imshow(rgb[0],cmap='plasma')
        plt.savefig(save_images_in + str(epoch)) 
        torch.save(net.state_dict(), pth_file_location)
