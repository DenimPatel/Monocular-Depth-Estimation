import sys
sys.path.append('../')

import torch
import torch.nn as nn

from utils.bilinear_sampler import apply_disparity

def reconstruct_using_disparity(image, disparity):
    """Reconstruct Image using Image and disparity from the approach of Bilinear Sampling

    Arguments:
        image {tensor} -- Image
        disparity {tensor} -- Disparity

    Returns:
        tensor -- Reconstructed Image
    """
    return apply_disparity(image, disparity)
