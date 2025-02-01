#!/usr/bin/env python3

import torch
from torch.nn import Conv2d

# 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())

# 检查cuDNN是否启用
print("cuDNN enabled:", torch.backends.cudnn.enabled)

# 创建一个卷积层并移动到GPU
conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3).cuda()

# 创建一个输入张量并移动到GPU
input_tensor = torch.randn(1, 3, 32, 32).cuda()

# 前向传播
output = conv(input_tensor)
print("Output shape:", output.shape)
