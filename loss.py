import torch
import torch.nn as nn
from bilinear_sampler import apply_disparity

def reconstruct_using_disparity(image, disparity):
    return apply_disparity(image, disparity)

