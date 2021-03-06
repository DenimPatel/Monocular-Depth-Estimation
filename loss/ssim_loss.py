import torch
import torch.nn as nn

def SSIM_loss(x, y):
    """Calculate Structural Similarity Score

    Arguments:
        x {tenosor} -- Image
        y {tensor} -- Image

    Returns:
        floatTensor -- SSIM loss : (1- SSIM)/2 between 0 to 1
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.functional.avg_pool2d(x, 3, 1, padding = 0)
    mu_y = nn.functional.avg_pool2d(y, 3, 1, padding = 0)

    sigma_x  = nn.functional.avg_pool2d(x ** 2, 3, 1, padding = 0) - mu_x ** 2
    sigma_y  = nn.functional.avg_pool2d(y ** 2, 3, 1, padding = 0) - mu_y ** 2

    sigma_xy = nn.functional.avg_pool2d(x * y , 3, 1, padding = 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)