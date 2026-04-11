import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import random

# 导入模型和数据集
from models.unet import UNet
from models.mkunet import ImprovedUNet, create_ablation_model, ABLATION_CONFIGS
from dataset import TIFSegmentationDataset

# ================= 配置区域 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_IMG_DIR = './dataset/kaggle_3m/test'
ABLATION_ROOT = './ablation_results'


def build_model_from_args(model_type, variant=None):
    """
    根据 model_type 和 variant 构建模型实例
    model_type: 'baseline' | 'innovation' | 'ablation'
    variant: 仅当 model_type='ablation' 时使用，如 'w/o_mkdc'
    """
    if model_type == 'baseline':
        mp = 'checkpoints/best_model_unet.pth'
        print("🚀 正在初始化基线模型 (Baseline U-Net)...")
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    elif model_type == 'innovation':
        mp = 'checkpoints/best_model_mkunet.pth'
        print("🚀 正在初始化完整创新模型 (Improved MK-UNet)...")
        model = ImprovedUNet(n_channels=3, n_classes=1).to(DEVICE)
    elif model_type == 'ablation':
        if variant not in ABLATION_CONFIGS:
            raise ValueError(f"未知消融变体: {variant}. 可选: {list(ABLATION_CONFIGS.keys())}")
        cfg = ABLATION_CONFIGS[variant]
        mp = os.path.join(ABLATION_ROOT, variant, 'checkpoints', 'best_model.pth')
        print(f"🚀 正在初始化消融变体模型: {cfg['name']}...")
        model = create_ablation_model(variant).to(DEVICE)
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    return model, mp


def calculate_metrics(pred, target, threshold=0.5):
    """计算单批次的 Dice 系数和 IoU"""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)

    return dice.item(), iou.item()


def evaluate_and_visualize(model, model_path, display_name, dataset, device, num_samples=6):
    print(f"\n🔍 正在加载权重文件: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {model_path}！")

    # 加载权重并开启推理模式
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ 权重加载成功！\n")

    # =========================================
    # 1. 核心改进：全测试集定量评估 (计算 Test Dice/IoU)
    # =========================================
    print("📊 正在对整个测试集进行定量指标评估，请稍候...")
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    test_dice_sum = 0.0
    test_iou_sum = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice, iou = calculate_metrics(outputs, masks)
            test_dice_sum += dice
            test_iou_sum += iou

    avg_test_dice = test_dice_sum / len(test_loader)
    avg_test_iou = test_iou_sum / len(test_loader)

    print("=" * 55)
    print(f"🏆 【{display_name}】最终测试集成绩单:")
    print(f"   Test - Dice: {avg_test_dice:.4f} | IoU: {avg_test_iou:.4f}")
    print("=" * 55)

    # =========================================
    # 2. 定性评估：生成可视化对比图
    # =========================================
    print(f"\n📸 正在抽取 {num_samples} 张样本生成【统一可视化大图】...")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))

    # 如果只有1张图，axes 不是二维数组
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row_idx, data_idx in enumerate(indices):
            image_tensor, mask_tensor = dataset[data_idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)

            pred_tensor = model(input_tensor)
            pred_binary = (pred_tensor.squeeze().cpu() > 0.5).float().numpy()

            img_vis = image_tensor.cpu() * std + mean
            img_vis = torch.clamp(img_vis, 0, 1).permute(1, 2, 0).numpy()
            mask_vis = mask_tensor.squeeze().cpu().numpy()

            axes[row_idx, 0].imshow(img_vis)
            axes[row_idx, 0].axis('off')
            if row_idx == 0:
                axes[row_idx, 0].set_title("Test Image (CLAHE)", fontsize=14, pad=10)

            axes[row_idx, 1].imshow(mask_vis, cmap='gray')
            axes[row_idx, 1].axis('off')
            if row_idx == 0:
                axes[row_idx, 1].set_title("Ground Truth", fontsize=14, pad=10)

            axes[row_idx, 2].imshow(pred_binary, cmap='gray')
            axes[row_idx, 2].axis('off')
            if row_idx == 0:
                axes[row_idx, 2].set_title(f"Prediction ({display_name})", fontsize=14, pad=10)

    plt.tight_layout()

    save_filename = f"{display_name.replace(' ', '_')}_predictions_grid.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n🎉 论文配图已成功保存为当前目录下的: 【{save_filename}】")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='模型预测与评估工具')
    parser.add_argument('--model_type', type=str, default='innovation',
                        choices=['baseline', 'innovation', 'ablation'],
                        help='模型类型')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_CONFIGS.keys()) if ABLATION_CONFIGS else None,
                        help='消融变体名称 (仅当 model_type=ablation 时需要)')
    parser.add_argument('--test_dir', type=str, default=TEST_IMG_DIR,
                        help='测试集目录')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='可视化样本数量')
    args = parser.parse_args()

    print(f"🚀 使用设备: {DEVICE}")
    print(f"📊 正在初始化测试数据集 (自带 CLAHE 处理)...")
    test_dataset = TIFSegmentationDataset(args.test_dir)

    # 构建模型
    model, model_path = build_model_from_args(args.model_type, args.variant)

    # 获取显示名称
    if args.model_type == 'baseline':
        display_name = 'Baseline'
    elif args.model_type == 'innovation':
        display_name = 'MK-UNet'
    else:
        display_name = ABLATION_CONFIGS[args.variant]['name']

    # 执行评估与可视化
    evaluate_and_visualize(model, model_path, display_name, test_dataset, DEVICE, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
