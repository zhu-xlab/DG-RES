import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from fft2d_functions import *


# 读取图像并进行傅里叶变换，分解图像为相位谱和幅值谱
image1 = image_read('./images/building_001.jpg', fixed_size=512)
image2 = image_read('./images/building_002.jpg', fixed_size=512)
image3 = image_read('./images/building_003.jpg', fixed_size=512)
image3 = image1 * 0.5 + image2 * 0.5
phase1, magnitude1 = image_fft(image1)
phase2, magnitude2 = image_fft(image2)
phase3, magnitude3 = image_fft(image3)

# alpha_phase = [0.5, 0.5]
# fused_phase = alpha_phase[0] * phase1 + alpha_phase[1] * phase2
# alpha_magnitude = [1.0, 0.0]
# fused_magnitude = alpha_magnitude[0] * magnitude1 + alpha_magnitude[1] * magnitude2

fused_phase = phase1 * 0.5 + phase2 * 0.5
fused_magnitude = magnitude1 * 0.5 + magnitude2 * 0.5

# fused_magnitude = magnitude_noise(fused_magnitude)
# fused_magnitude = remove_high_frequency(fused_magnitude, is_shift=True, radius=16)
# fused_magnitude = remove_low_frequency(fused_magnitude, is_shift=True, radius=2)
# fused_magnitude = random_drop_frequency(fused_magnitude, p=0.5)
# fused_magnitude = histogram_equalization_2d(fused_magnitude.unsqueeze(dim=0)).squeeze(dim=0)
# fused_magnitude = random_resize_center_symmetric(fused_magnitude, is_shift=True)
# fused_magnitude = remove_low_frequency(fused_magnitude, is_shift=True, radius=5)*1.0 \
#                 + remove_high_frequency(fused_magnitude, is_shift=True, radius=5)*0.5

# 逆傅里叶变换，将图像重构
fused_image = image_ifft(fused_phase, fused_magnitude).clip(max=255, min=0)

#  将边缘移至中心，便于显示
magnitude1 = fft.fftshift(magnitude1)
magnitude2 = fft.fftshift(magnitude2)
magnitude3 = fft.fftshift(magnitude3)
fused_magnitude = fft.fftshift(fused_magnitude)

# 归一化
magnitude1 = magnitude_norm(magnitude1)
magnitude2 = magnitude_norm(magnitude2)
magnitude3 = magnitude_norm(magnitude3)
fused_magnitude = magnitude_norm(fused_magnitude)

# magnitude1 = fft.ifftshift(magnitude1)
# magnitude2 = fft.ifftshift(magnitude2)
# fused_magnitude = fft.ifftshift(fused_magnitude)

# 显示原始图像和重构图像
plt.figure(figsize=(12, 6))

plt.subplot(241)
plt.imshow(image1.permute(1, 2, 0))
plt.title('Image 1')
plt.axis('off')

plt.subplot(245)
plt.imshow(magnitude1.permute(1, 2, 0))
plt.title('Magnitude Spectrum 1')
plt.axis('off')

plt.subplot(242)
plt.imshow(image2.permute(1, 2, 0))
plt.title('Image 2')
plt.axis('off')

plt.subplot(246)
plt.imshow(magnitude2.permute(1, 2, 0))
plt.title('Magnitude Spectrum 2')
plt.axis('off')

plt.subplot(243)
plt.imshow(image3.permute(1, 2, 0))
plt.title('Image 3')
plt.axis('off')

plt.subplot(247)
plt.imshow(magnitude3.permute(1, 2, 0))
plt.title('Magnitude Spectrum 3')
plt.axis('off')

plt.subplot(244)
plt.imshow(fused_image.permute(1, 2, 0))
plt.title('Fused Image')
plt.axis('off')

plt.subplot(248)
plt.imshow(fused_magnitude.permute(1, 2, 0))
plt.title('Fused Magnitude Spectrum')
plt.axis('off')

plt.show()

