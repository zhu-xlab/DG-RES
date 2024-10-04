# from PIL import Image, ImageFilter
# import matplotlib.pyplot as plt


# # 读取图像
# img = Image.open("./images/dog_art_001.jpg")

# # 对图像进行模糊处理
# blurred_img = img.filter(ImageFilter.GaussianBlur(radius=1))  # 调整 radius 控制模糊程度

# # 显示模糊后的图像
# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(blurred_img)
# plt.title('Blurred Image')
# plt.axis('off')

# plt.tight_layout()
# plt.show()


import torch
import matplotlib.pyplot as plt

# 假设有一组 C 个数据，表示为一个 PyTorch 张量
data = torch.tensor([0.11, 0.13, 0.85, 0.76, 0.82, 0.9, 1.0])  # 举例：假设这些数据分布在 0 到 1 之间

# 对数据进行排序并获取排名
sorted_data, indices = torch.sort(data)
rank = torch.linspace(0, 1, len(sorted_data))  # 数据在原分布中的排名
cdf_values = (indices.float() + 1) / len(sorted_data)  # 原分布的累积分布函数值
transformed_data = torch.lerp(torch.zeros_like(sorted_data), torch.ones_like(sorted_data), cdf_values)

# 绘制前后数据的分布情况
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(range(len(data)), data, color='blue', label='原始数据')
plt.title('原始数据分布')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(transformed_data)), transformed_data, color='red', label='校正后的数据')
plt.title('校正后数据分布')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

# import torch

# x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# # 计算 L2 范数
# norm_value = torch.norm(x, dim=-1).mean()
# print(norm_value)