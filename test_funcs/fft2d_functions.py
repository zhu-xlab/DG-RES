import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

def image_read(image_path, fixed_size):
    transform = transforms.Compose([transforms.Resize((fixed_size, fixed_size)), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")  
    image = transform(image)
    return image

def image_fft(image):
    # 执行傅里叶变换和幅值谱/相位谱计算（分别对两张图像执行）
    fft_image = fft.fft2(image, dim=(-2, -1))
    magnitude = torch.abs(fft_image)
    phase = torch.angle(fft_image)
    return phase, magnitude

def image_ifft(phase, magnitude):
    return fft.ifft2(magnitude * torch.exp(1j * phase)).real

def magnitude_norm(magnitude):
    min_magnitude = magnitude.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
    max_magnitude = magnitude.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
    magnitude = (magnitude - min_magnitude) / (max_magnitude - min_magnitude)
    magnitude *= 255                    
    return magnitude

def magnitude_noise(magnitude):
    channels, height, width = magnitude.size()
    white_noise = torch.randn(magnitude.size())
    white_noise = torch.randn((channels, height, width))
    scaled_white_noise = (1 + 0.5 * white_noise).clip(min=0)
    noised_magnitude = scaled_white_noise*magnitude
    return noised_magnitude

def remove_high_frequency(input_tensor, is_shift=True, radius=10):
    if is_shift:
        input_tensor = fft.fftshift(input_tensor)
    # Ensure the input tensor has three dimensions (BxHxW)
    assert len(input_tensor.shape) == 3, "Input tensor must have three dimensions (BxHxW)"
    
    # Get the shape of the input tensor
    B, H, W = input_tensor.shape
    
    # Create a binary mask with ones in the center and zeros outside
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius**2
    mask = mask.float().unsqueeze(0)  # Add a batch dimension

    # Apply the mask to each element in the batch
    masked_tensor = input_tensor * mask
    if is_shift:
        masked_tensor = fft.ifftshift(masked_tensor)
    return masked_tensor

def remove_low_frequency(input_tensor, is_shift=True, radius=10):
    if is_shift:
        input_tensor = fft.fftshift(input_tensor)
    # Ensure the input tensor has three dimensions (BxHxW)
    assert len(input_tensor.shape) == 3, "Input tensor must have three dimensions (BxHxW)"
    
    # Get the shape of the input tensor
    B, H, W = input_tensor.shape
    
    # Create a binary mask with ones in the center and zeros outside
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) >= radius**2
    mask = mask.float().unsqueeze(0)  # Add a batch dimension

    # Apply the mask to each element in the batch
    masked_tensor = input_tensor * mask
    if is_shift:
        masked_tensor = fft.ifftshift(masked_tensor)
    return masked_tensor

def random_drop_frequency(magnitude, p=0.5):
    # 获取图像尺寸
    height, width = magnitude.shape[-2:]

    # 随机选择30%的频率，并将其幅度置零
    num_freq_to_remove = int(p * height * width)
    indices_to_remove = torch.randperm(height * width)[:num_freq_to_remove]
    mask = torch.ones(height, width)
    mask[indices_to_remove // width, indices_to_remove % width] = 0

    # 将选择的频率对应的幅度置零
    magnitude = magnitude * mask
    return magnitude

def histogram_equalization_2d(magnitude):
    def equalize_channel(channel):
        # 计算直方图
        hist = torch.histc(channel, bins=256, min=0, max=255)

        # 计算累积分布函数
        cdf = hist.cumsum(0)

        # 归一化到 0-1 范围
        cdf = cdf / cdf[-1]

        # 将原始通道映射到均衡化的通道
        equalized_channel = torch.clamp((cdf[channel.int()] * 255).round(), 0, 255).byte()

        return equalized_channel

    B, C, H, W = magnitude.shape

    # 将张量展平为二维（B * C, H * W）
    flattened_tensor = magnitude.view(B * C, H * W)
    flattened_tensor_min = flattened_tensor.min(dim=1, keepdim=True)[0]
    flattened_tensor_max = flattened_tensor.max(dim=1, keepdim=True)[0]
    flattened_tensor = (flattened_tensor - flattened_tensor_min) / (flattened_tensor_max - flattened_tensor_min)
    flattened_tensor = (flattened_tensor*255).float()

    # 对每个通道的二维张量进行直方图均衡化
    equalized_tensor = torch.stack([equalize_channel(channel.view(1, -1)).view(-1) for channel in flattened_tensor])
    equalized_tensor = equalized_tensor / 255.0
    equalized_tensor = equalized_tensor*(flattened_tensor_max - flattened_tensor_min) + flattened_tensor_min

    # 将结果重新整形为原始形状
    equalized_tensor = equalized_tensor.view(B, C, H, W)
    return equalized_tensor

def random_resize_center_symmetric(tensor, is_shift=False, scale_range=(0.9, 1.1)):
    if is_shift:
        tensor = fft.fftshift(tensor)

    if  tensor.dim() == 3:
        tensor = tensor.unsqueeze(dim=0)
    B, C, H, W = tensor.size()

    # randomly generate scale factor
    scale_factor = torch.rand(B) * (scale_range[1] - scale_range[0]) + scale_range[0]
    print (scale_factor)
    new_H = (H * scale_factor).int()
    new_W = (W * scale_factor).int()

    # zoom in
    if scale_factor >= 1.0:
        resized_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)        
        crop_top = (new_H - H) // 2
        crop_bottom = crop_top + H 
        crop_left = (new_W - W) // 2
        crop_right = crop_left + W
        resized_tensor = resized_tensor[:, :, crop_top:crop_bottom, crop_left:crop_right]
    # zoom out
    else:
        resized_tensor = torch.zeros_like(tensor)
        center_tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
        crop_top = (H - new_H) // 2
        crop_bottom = crop_top + new_H 
        crop_left = (W - new_W) // 2
        crop_right = crop_left + new_W
        resized_tensor[:, :, crop_top:crop_bottom, crop_left:crop_right] = center_tensor

    if is_shift:
        resized_tensor = fft.ifftshift(resized_tensor.squeeze(dim=0))
    return resized_tensor.squeeze(dim=0)