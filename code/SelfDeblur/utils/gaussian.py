from skimage import io, img_as_float
from skimage.filters import gaussian
import numpy as np
from PIL import Image

# 读取图像并转换为浮点数类型
image = img_as_float(io.imread('datasets/test/exmp_kernel5_img.png'))

# 应用高斯模糊
gaussian_blurred = gaussian(image, sigma=1)

# 将浮点图像数据标准化到0-255范围
normalized_img = (gaussian_blurred - np.min(gaussian_blurred)) / (np.max(gaussian_blurred) - np.min(gaussian_blurred)) * 255
# 转换为uint8类型
uint8_img = normalized_img.astype(np.uint8)

# 创建PIL图像，使用'L'模式因为在处理灰度图像
img = Image.fromarray(uint8_img, 'L')

# 保存图像
img.save('gaussian_kernel5_img.png')
