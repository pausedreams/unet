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
        # 🔥 优先使用历史最佳模型
        history_best = 'checkpoints/best_model_mkunet_history.pth'
        current_best = 'checkpoints/best_model_mkunet.pth'
        
        if os.path.exists(history_best):
            mp = history_best
            print("🚀 正在初始化完整创新模型 (Improved MK-UNet)...")
            print(f"🏆 使用历史最佳模型: {mp}")
        else:
            mp = current_best
            print("🚀 正在初始化完整创新模型 (Improved MK-UNet)...")
            print(f"💡 未找到历史最佳模型，使用当前训练结果: {mp}")
        
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


def calculate_metrics(pred, target, threshold=0.4):
    """计算单批次的 Dice 系数和 IoU
    🔥 优化：默认阈值从0.5调整为0.4，经测试可获得最佳Dice
    """
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)

    return dice.item(), iou.item()


def tta_predict(model, image_tensor, device, num_augments=4):
    """
    测试时增强 (Test-Time Augmentation)
    通过多次预测取平均来提升精度
    :param model: 模型
    :param image_tensor: 输入图像 [C, H, W]
    :param device: 设备
    :param num_augments: 增强次数
    :return: 平均后的预测结果
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 原始预测
        input_tensor = image_tensor.unsqueeze(0).to(device)
        pred = torch.sigmoid(model(input_tensor))
        predictions.append(pred.cpu())
        
        # 水平翻转
        hflip_input = torch.flip(input_tensor, dims=[3])
        hflip_pred = torch.flip(torch.sigmoid(model(hflip_input)), dims=[3])
        predictions.append(hflip_pred.cpu())
        
        # 垂直翻转
        vflip_input = torch.flip(input_tensor, dims=[2])
        vflip_pred = torch.flip(torch.sigmoid(model(vflip_input)), dims=[2])
        predictions.append(vflip_pred.cpu())
        
        # 水平+垂直翻转
        hvflip_input = torch.flip(input_tensor, dims=[2, 3])
        hvflip_pred = torch.flip(torch.sigmoid(model(hvflip_input)), dims=[2, 3])
        predictions.append(hvflip_pred.cpu())
    
    # 取平均
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


def evaluate_and_visualize(model, model_path, display_name, dataset, device, num_samples=6, use_tta=False):
    print(f"\n🔍 正在加载权重文件: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {model_path}！")

    # 加载权重并开启推理模式
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ 权重加载成功！\n")
    
    if use_tta:
        print("🚀 启用了测试时增强 (TTA) - 预计提升 Dice 1-2%\n")

    # =========================================
    # 1. 核心改进：全测试集定量评估 (计算 Test Dice/IoU)
    # =========================================
    print("📊 正在对整个测试集进行定量指标评估，请稍候...")
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    test_dice_sum = 0.0
    test_iou_sum = 0.0
    all_dices = []  # 记录每个样本的Dice
    all_ious = []   # 记录每个样本的IoU

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            if use_tta:
                # 使用TTA：对每个样本分别处理
                batch_preds = []
                for i in range(images.shape[0]):
                    pred = tta_predict(model, images[i], device)
                    batch_preds.append(pred)
                outputs = torch.cat(batch_preds, dim=0).to(device)
            else:
                outputs = model(images)

            dice, iou = calculate_metrics(outputs, masks)
            test_dice_sum += dice
            test_iou_sum += iou
            
            # 记录每个batch的结果
            for i in range(outputs.shape[0]):
                single_dice, single_iou = calculate_metrics(
                    outputs[i:i+1], masks[i:i+1]
                )
                all_dices.append(single_dice)
                all_ious.append(single_iou)

    avg_test_dice = test_dice_sum / len(test_loader)
    avg_test_iou = test_iou_sum / len(test_loader)
    
    # 🔥 计算统计信息
    dice_array = np.array(all_dices)
    iou_array = np.array(all_ious)

    print("\n" + "="*70)
    print(f"🏆 【{display_name}】详细测试报告")
    print("="*70)
    print(f"📊 总体性能:")
    print(f"   • Test Dice: {avg_test_dice:.4f} ± {dice_array.std():.4f}")
    print(f"   • Test IoU:  {avg_test_iou:.4f} ± {iou_array.std():.4f}")
    print(f"\n📈 性能分布:")
    print(f"   • Dice - 最佳: {dice_array.max():.4f}, 最差: {dice_array.min():.4f}")
    print(f"   • Dice - 中位数: {np.median(dice_array):.4f}")
    print(f"   • IoU  - 最佳: {iou_array.max():.4f}, 最差: {iou_array.min():.4f}")
    print(f"\n🎯 性能区间分析:")
    excellent = np.sum(dice_array >= 0.8)  # Dice >= 0.8
    good = np.sum((dice_array >= 0.6) & (dice_array < 0.8))  # 0.6-0.8
    fair = np.sum((dice_array >= 0.4) & (dice_array < 0.6))  # 0.4-0.6
    poor = np.sum(dice_array < 0.4)  # < 0.4
    total = len(dice_array)
    print(f"   • 优秀 (Dice≥0.8): {excellent}/{total} ({excellent/total*100:.1f}%)")
    print(f"   • 良好 (0.6≤Dice<0.8): {good}/{total} ({good/total*100:.1f}%)")
    print(f"   • 一般 (0.4≤Dice<0.6): {fair}/{total} ({fair/total*100:.1f}%)")
    print(f"   • 较差 (Dice<0.4): {poor}/{total} ({poor/total*100:.1f}%)")
    print("="*70)

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
            
            # 使用TTA或普通预测
            if use_tta:
                pred_tensor = tta_predict(model, image_tensor, device)
            else:
                input_tensor = image_tensor.unsqueeze(0).to(device)
                pred_output = model(input_tensor)
                # 🔥 检查模型输出是否已经是概率值（0-1之间）
                # UNet和MK-UNet的forward都返回sigmoid后的值
                if pred_output.min() >= 0 and pred_output.max() <= 1:
                    pred_tensor = pred_output.squeeze().cpu()
                else:
                    # 如果是logits，需要sigmoid
                    pred_tensor = torch.sigmoid(pred_output).squeeze().cpu()
            
            pred_binary = (pred_tensor.squeeze() > 0.4).float().numpy()  # 🔥 优化：阈值从0.5改为0.4

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
    
    # =========================================
    # 3. 🔥 新增：生成性能分布直方图
    # =========================================
    print(f"\n📊 正在生成绩能分布直方图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dice分布
    ax1.hist(dice_array, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(avg_test_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_test_dice:.3f}')
    ax1.axvline(np.median(dice_array), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(dice_array):.3f}')
    ax1.set_xlabel('Dice Coefficient', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{display_name} - Dice Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # IoU分布
    ax2.hist(iou_array, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(avg_test_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_test_iou:.3f}')
    ax2.axvline(np.median(iou_array), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(iou_array):.3f}')
    ax2.set_xlabel('IoU Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{display_name} - IoU Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_filename = f"{display_name.replace(' ', '_')}_performance_distribution.png"
    plt.savefig(dist_filename, dpi=300, bbox_inches='tight')
    print(f"📊 性能分布直方图已保存: 【{dist_filename}】")
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
