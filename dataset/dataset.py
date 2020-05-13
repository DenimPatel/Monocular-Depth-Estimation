import os
import time

import numpy as np

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from params import *

class KITTIDataset_from_txt(Dataset):
    """Generate Pytorch ready KittiDataset using txt file containing image

    Arguments:
        Dataset {pytorch dataset} -- superclass 
    """
    def __init__(self):
        self.rootdir = train_dataset_location
        # "/home/baladhurgesh97/Dataset/" # directory where dataset is present physically
        # /Users/denimpatel/Desktop/Deep Learning/Monocular-depth-estimation/2011_09_26
        self.extension_of_images = image_type # use ".jpg" or ".png" as per dataset
        self.sanity_check_images = sanity_check_images # True will check each image is valid, considerable computation added
        self.left_images = []
        self.right_images = []
        self.cols = 512
        self.rows = 256
        self.no_of_images = 0

        # read file and store all the images 
        print("Loading Dataset: from -> ", self.rootdir)
        start = time.time()
        with open(TRAIN_DATASET_TXT, mode = 'r') as csv_file:
            for line in csv_file:
                stereo_img = line.split(' ') # each line contains two image names -> "left image & right image"
                left_image_name = os.path.join(self.rootdir, stereo_img[0][:-4]+self.extension_of_images)
                right_image_name = os.path.join(self.rootdir, stereo_img[1][:-4]+self.extension_of_images)
                # print("left image name = ", left_image_name)
                if not self.sanity_check_images:
                    self.left_images.append(left_image_name)
                    self.right_images.append(right_image_name)
                    self.no_of_images += 1
                else:
                    left_img = cv2.imread(left_image_name)
                    right_img = cv2.imread(right_image_name)
                    if left_img is not None and right_img is not None:
                        self.left_images.append(left_image_name)
                        self.right_images.append(right_image_name)
                        self.no_of_images += 1
                    else:
                        print("Error found at : " , left_image_name)
                        print("Error found at : ", right_image_name)

        print("Loading Dataset: COMPLETE! took ", time.time()-start, " seconds")
        print("Total Stereo images acquired: ", self.no_of_images)       

    def __len__(self):
        return self.no_of_images

    def __getitem__(self, idx):
        # read stereo images from disk
        if idx == 0:
            left_img = cv2.imread(self.left_images[idx])
            right_img = cv2.imread(self.right_images[idx])
        else:
            left_img = cv2.imread(self.left_images[idx])
            right_img = cv2.imread(self.right_images[idx])

        np.asarray(left_img)
        np.asarray(right_img)

        # resize as per model requirement
        left_img = cv2.resize(left_img,(self.cols, self.rows))
        right_img = cv2.resize(right_img,(self.cols, self.rows))

        # reshape for pytorch [channels, rows, cols]
        left_img = np.moveaxis(left_img, 2,0)
        right_img = np.moveaxis(right_img, 2,0)
        return {"left_img":left_img,"right_img":right_img,"prev_left_img":prev_left_img,"prev_right_img":prev_right_img}

class KITTIDataset_from_folder(Dataset):
    """Generate Pytorch ready KittiDataset from dataset of images

    Arguments:
        Dataset {pytorch dataset} -- superclass 
    """
    def __init__(self):
        self.rootdir = train_dataset_location
        # "/home/baladhurgesh97/Dataset/" # directory where dataset is present physically
        # /Users/denimpatel/Desktop/Deep Learning/Monocular-depth-estimation/2011_09_26
        self.extension_of_images = image_type # use ".jpg" or ".png" as per dataset
        self.sanity_check_images = sanity_check_images
        """
        @TODO: implement Sanity check for this method
        """
        self.left_images = []
        self.right_images = []
        self.cols = 512
        self.rows = 256
        self.total_left_images = 0
        self.total_right_images = 0
        print("Loading Dataset: from -> ", self.rootdir)
        start = time.time()
        for subdir, dirs, files in os.walk(self.rootdir):
            if "image_02/data" in subdir: #Left RGB Folder
                for file in files:
                    if ".png" in file or ".jpg" in file:
                        # if not ("0000.jpg" or "0000.png") in file:
                        left_file = os.path.join(subdir, file)
                        self.left_images.append(left_file)
                        self.total_left_images += 1
                
            if "image_03/data" in subdir: #Right RGB Folder
                for file in files:
                    if ".png" in file or ".jpg" in file:
                        # if not ("0000.jpg" or "0000.png") in file:
                        right_file = os.path.join(subdir, file)
                        self.right_images.append(right_file)
                        self.total_right_images += 1
        self.left_images.sort()
        self.right_images.sort()
        assert(self.total_left_images == self.total_right_images)
        print("Loading Dataset: COMPLETE! took ", time.time()-start, " seconds")
        print("Total Stereo images acquired: ", len(self.right_images))       

    def __len__(self):
        return len(self.right_images)

    def __getitem__(self, idx):
        # splits = self.left_images[idx].split("/")
        # numbers = int(splits[-1][:-4])
        # path = self.left_images[idx].split("000")
        
        # read stereo images from disk
        if idx == 0:
            left_img = cv2.imread(self.left_images[idx])
            right_img = cv2.imread(self.right_images[idx])
        else:
            left_img = cv2.imread(self.left_images[idx])
            right_img = cv2.imread(self.right_images[idx])

        np.asarray(left_img)
        np.asarray(right_img)

        # resize as per model requirement
        left_img = cv2.resize(left_img,(self.cols, self.rows))
        right_img = cv2.resize(right_img,(self.cols, self.rows))

        # reshape for pytorch [channels, rows, cols]
        left_img = np.moveaxis(left_img, 2,0)
        right_img = np.moveaxis(right_img, 2,0)

        return {"left_img":left_img,"right_img":right_img}

class KITTIDataset_from_txt_for_temporal_smoothness(Dataset):
    """Generate Pytorch ready KittiDataset using txt file containing image

    Arguments:
        Dataset {pytorch dataset} -- superclass 
    """
    def __init__(self):
        self.rootdir = train_dataset_location
        # "/home/baladhurgesh97/Dataset/" # directory where dataset is present physically
        # /Users/denimpatel/Desktop/Deep Learning/Monocular-depth-estimation/2011_09_26
        self.extension_of_images = image_type # use ".jpg" or ".png" as per dataset
        self.sanity_check_images = sanity_check_images # True will check each image is valid, considerable computation added
        self.left_images = []
        self.right_images = []
        self.cols = 512
        self.rows = 256
        self.no_of_images = 0

        # read file and store all the images 
        print("Loading Dataset: from -> ", self.rootdir)
        start = time.time()
        with open(TRAIN_DATASET_TXT, mode = 'r') as csv_file:
            for line in csv_file:
                stereo_img = line.split(' ') # each line contains two image names -> "left image & right image"
                left_image_name = os.path.join(self.rootdir, stereo_img[0][:-4]+self.extension_of_images)
                right_image_name = os.path.join(self.rootdir, stereo_img[1][:-4]+self.extension_of_images)
                # print("left image name = ", left_image_name)
                if not self.sanity_check_images:
                    self.left_images.append(left_image_name)
                    self.right_images.append(right_image_name)
                    self.no_of_images += 1
                else:
                    left_img = cv2.imread(left_image_name)
                    right_img = cv2.imread(right_image_name)
                    if left_img is not None and right_img is not None:
                        self.left_images.append(left_image_name)
                        self.right_images.append(right_image_name)
                        self.no_of_images += 1
                    else:
                        pass
                        print("Error found at : " , left_image_name)
                        print("Error found at : ", right_image_name)

        print("Loading Dataset: COMPLETE! took ", time.time()-start, " seconds")
        print("Total Stereo images acquired: ", self.no_of_images)       

    def __len__(self):
        return self.no_of_images

    def __getitem__(self, idx):
        # read stereo images from disk
        left_img = cv2.imread(self.left_images[idx])
        right_img = cv2.imread(self.right_images[idx])

        np.asarray(left_img)
        np.asarray(right_img)
        
        # resize as per model requirement
        left_img = cv2.resize(left_img,(self.cols, self.rows))
        right_img = cv2.resize(right_img,(self.cols, self.rows))

        # reshape for pytorch [channels, rows, cols]
        left_img = np.moveaxis(left_img, 2,0)
        right_img = np.moveaxis(right_img, 2,0)

        return {"left_img":left_img,"right_img":right_img}

class KITTIDataset_from_folder_for_temporal_smoothness(Dataset):
    """Generate Pytorch ready KittiDataset from dataset of images

    Arguments:
        Dataset {pytorch dataset} -- superclass 
    """
    def __init__(self):
        self.rootdir = train_dataset_location
        self.extension_of_images = image_type # use ".jpg" or ".png" as per dataset
        self.sanity_check_images = sanity_check_images
        """
        @TODO: implement Sanity check for this method
        """
        self.left_images = []
        self.right_images = []
        self.cols = 512
        self.rows = 256
        self.total_left_images = 0
        self.total_right_images = 0
        print("Loading Dataset: from -> ", self.rootdir)
        start = time.time()
        for subdir, dirs, files in os.walk(self.rootdir):
            if "image_02/data" in subdir: #Left RGB Folder
                for file in files:
                    if ".png" in file or ".jpg" in file:
                        # if not ("0000.jpg" or "0000.png") in file:
                        left_file = os.path.join(subdir, file)
                        self.left_images.append(left_file)
                        self.total_left_images += 1
                
            if "image_03/data" in subdir: #Right RGB Folder
                for file in files:
                    if ".png" in file or ".jpg" in file:
                        # if not ("0000.jpg" or "0000.png") in file:
                        right_file = os.path.join(subdir, file)
                        self.right_images.append(right_file)
                        self.total_right_images += 1
        self.left_images.sort()
        self.right_images.sort()
        assert(self.total_left_images == self.total_right_images)
        print("Loading Dataset: COMPLETE! took ", time.time()-start, " seconds")
        print("Total Stereo images acquired: ", len(self.right_images))       

    def __len__(self):
        return len(self.right_images)

    def __getitem__(self, idx):
        # read stereo images from disk
        if idx == 0:
            left_img = cv2.imread(self.left_images[idx])
            prev_left_img = cv2.imread(self.left_images[idx])
            right_img = cv2.imread(self.right_images[idx])
            prev_right_img = cv2.imread(self.right_images[idx])
        else:
            left_img = cv2.imread(self.left_images[idx])
            prev_left_img = cv2.imread(self.left_images[idx-1])
            right_img = cv2.imread(self.right_images[idx])
            prev_right_img = cv2.imread(self.right_images[idx-1])

        np.asarray(left_img)
        np.asarray(prev_left_img)
        np.asarray(right_img)
        np.asarray(prev_right_img)
        # resize as per model requirement
        left_img = cv2.resize(left_img,(self.cols, self.rows))
        prev_left_img = cv2.resize(prev_left_img,(self.cols, self.rows))
        right_img = cv2.resize(right_img,(self.cols, self.rows))
        prev_right_img = cv2.resize(prev_right_img,(self.cols, self.rows))
        # reshape for pytorch [channels, rows, cols]
        left_img = np.moveaxis(left_img, 2,0)
        prev_left_img = np.moveaxis(prev_left_img, 2,0)
        right_img = np.moveaxis(right_img, 2,0)
        prev_right_img = np.moveaxis(prev_right_img, 2,0)

        return {"left_img":left_img, "right_img":right_img, "prev_left_img":prev_left_img, "prev_right_img":prev_right_img}
