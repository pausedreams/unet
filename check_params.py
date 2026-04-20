from models.mkunet import ImprovedUNet
import torch

model = ImprovedUNet(n_channels=3, n_classes=1)
total = sum(p.numel() for p in model.parameters())

print(f'总参数量: {total:,} ({total/1e6:.4f}M)')
print(f'\n各模块参数量分布:')

for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f'  {name}: {params:,} ({params/total*100:.1f}%)')

# 对比标准 U-Net
from models.unet import UNet
unet = UNet(n_channels=3, n_classes=1)
unet_params = sum(p.numel() for p in unet.parameters())
print(f'\n对比: 标准 U-Net 参数量: {unet_params:,} ({unet_params/1e6:.2f}M)')
print(f'MK-UNet 减少了: {(1 - total/unet_params)*100:.1f}%')
