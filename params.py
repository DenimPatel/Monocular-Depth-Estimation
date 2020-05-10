LEARNING_RATE = 0.0001
resume_trining = True
BATCH_SIZE = 4
MANY_WORKERS = 8
EPOCH = 50
"""
weights for losses
"""
appearance_matching_loss_weight = 1.0
LR_loss_weight = 0.05
disparity_smoothness_loss_weight = 0.05
alpha_appearance_matching_loss = 0.3
temporal_disparity_smoothness_loss_weight = 0.05

"""
GPU vs CPU
"""
is_gpu_available = False
compute_device = "cpu:0" # "cpu:0" or "cuda:0"
if is_gpu_available:
    compute_device = "cuda:0"
    print("USING GPU AS A COMPUTE DEVICE")
else:
    print("USING CPU AS A COMPUTE DEVICE")


"""
dataset 
"""
# dataset_location = "/home/denimpatel2020/Monocular-Depth-Estimation/"
train_dataset_location = "/home/bala/DeepLearning/Train_Data/"
test_dataset_location = "/home/bala/DeepLearning/Train_Data/"


image_type = ".jpg" #or ".png"
sanity_check_images = False 

method_of_training = "folder" # "eigen_split" or "folder"

"""
@TODO: Two pth file for read and write  
"""
pth_file_location = "/home/bala/DeepLearning/Monocular-Depth-Estimation-master/wts_monodepth_gcp2.pth"
save_images_in = "/home/bala/DeepLearning/Monocular-Depth-Estimation-master/epoch/"
# save_images_in = '/home/baladhurgesh97/wts_monodepth_gcp1.pth'

"""
Test
"""
width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

min_depth = 1e-3
max_depth = 80

save_test_images_in = "/home/bala/DeepLearning/Monocular-Depth-Estimation-master/epoch/"
save_statistics_in = "/home/bala/DeepLearning/Monocular-Depth-Estimation-master/epoch/" 
