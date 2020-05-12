import torch
import torch.nn as nn

def gradient_x(image):
    """Find gradient along x

    Arguments:
        image {tensor} -- image

    Returns:
        tensor -- dx(image)
    """
    return image[:,:,:,:-1]-image[:,:,:,1:]

def gradient_y(image):
    """Find gradient along y

    Arguments:
        image {tensor} -- image

    Returns:
        tensor -- dy(image)
    """
    return image[:,:,:-1,:]-image[:,:,1:,:]

def gradient_t(prev_image, image):
    """Find gradient along time: time derivative using two images

    Arguments:
        prev_image {tensor} -- image at timestamp 1
        image {tensor} -- image at timestamp 2

    Returns:
        tensor -- time derivative of image
    """
    return prev_image-image

def disparity_smoothness(image, disparity):
    """Calculate "Edge aware" Disparity Smoothness loss 

    Arguments:
        image {tensor} -- image
        disparity {tensor} -- disparity

    Returns:
        FloatTensor -- loss 
    """
    grad_img_x = [gradient_x(i) for i in image]
    grad_img_y = [gradient_y(i) for i in image]

    grad_disp_x = [gradient_x(i) for i in disparity]
    grad_disp_y = [gradient_y(i) for i in disparity]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in grad_img_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in grad_img_y]

    smoothness_x = [grad_disp_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [grad_disp_y[i] * weights_y[i] for i in range(4)]
    
    smoothness_x = [torch.nn.functional.pad(k,(0,1,0,0,0,0,0,0),mode='constant') for k in smoothness_x]
    smoothness_y = [torch.nn.functional.pad(k,(0,0,0,1,0,0,0,0),mode='constant') for k in smoothness_y]

    disp_smoothness = smoothness_x + smoothness_y

    disp_loss = [torch.mean(torch.abs(disp_smoothness[i])) / 2 ** i for i in range(4)]
    return disp_loss

def temporal_disparity_smoothness(prev_image, image, prev_disparity, disparity):
    """Calculate Temporal Smoothness loss

    Arguments:
        prev_image {tensor} -- Image at timestamp 1
        image {tensor} -- Image at timestamp 2
        prev_disparity {tensro} -- Disparity at timestamp 1
        disparity {tensor} -- Disparity at timestamp 2

    Returns:
        FloatTensor -- loss
    """
    grad_img_t = [gradient_t(prev_image[i],image[i]) for i in range(4)]

    grad_disp_t = [gradient_t(prev_disparity[i],disparity[i]) for i in range(4)]

    weights_t = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in grad_img_t]

    smoothness_t = [grad_disp_t[i] * weights_t[i] for i in range(4)]

    temporal_disp_loss = [torch.mean(torch.abs(smoothness_t[i])) / 2 ** i for i in range(4)]
    
    return temporal_disp_loss