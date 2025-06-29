# ✅

import torch
print(f"PyTorch版本: {torch.__version__}")  # 应 ≥ 2.0.0
print(f"MPS可用: {torch.backends.mps.is_available()}")  # 应返回 True
print(f"当前设备: {torch.device('mps')}")  # 应显示 'mps'
