import torch
import torch.nn as nn

def scale_pyramid(img, num_scales):
    """Generate Image pyamid

    Arguments:
        img {tensor} -- Base Image
        num_scales {tensor} -- Number of levels required in pyramid of Image

    Returns:
        tensor -- pyramid of Image
    """
    scaled_imgs = [img]
    s = img.size()
    h = int(s[2])
    w = int(s[3])
    for i in range(num_scales-1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        temp = nn.functional.upsample(img, [nh, nw], mode='nearest')
        scaled_imgs.append(temp)
    return scaled_imgs