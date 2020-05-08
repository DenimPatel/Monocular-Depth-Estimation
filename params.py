resume_trining = False
BATCH_SIZE = 1
MANY_WORKERS = 0

"""
weights for losses
"""
appearance_matching_loss_weight = 1.0
LR_loss_weight = 0.0
disparity_smoothness_loss_weight = 0.1
alpha_appearance_matching_loss = 0.3

"""
GPU vs CPU
"""
is_gpu_available = False
compute_device = "cpu:0" # "cpu:0" or "cuda:0"
if is_gpu_available:
    compute_device = "cuda:0"

"""
dataset 
"""
dataset_location = "/Users/denimpatel/Desktop/Deep Learning/Monocular-depth-estimation/"
# dataset_location = "/home/baladhurgesh97/Dataset/"

image_type = ".png" #or ".jpg"
sanity_check_images = False 

method_of_training = "folder" # "eigen_split" or "folder"

pth_file_location = '/home/baladhurgesh97/wts_monodepth_gcp1.pth'
save_images_in = '/Users/denimpatel/Desktop/Deep Learning/Monocular-depth-estimation/epoch/'
# save_images_in = '/home/baladhurgesh97/wts_monodepth_gcp1.pth'


LEARNING_RATE = 0.0001