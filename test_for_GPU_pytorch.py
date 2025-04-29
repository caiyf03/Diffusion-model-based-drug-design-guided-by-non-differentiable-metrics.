import torch
print(torch.cuda.is_available())  # 如果返回 True，说明支持 GPU
print(torch.version.cuda)         # 如果安装了 GPU 版本，会显示 CUDA 版本（如 10.2）
print(torch.__version__)          # 显示 PyTorch 版本（如 1.12.1）