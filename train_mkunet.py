import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import swanlab

# --- 导入你的改进版模型和数据集 ---
from models.mkunet import ImprovedUNet 
from dataset import TIFSegmentationDataset  # 改为TIF数据集

# ================= 配置区域 =================
# 训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 60  # 🔥 增加训练轮数到 60
LEARNING_RATE = 3e-4  # 🔥 降低学习率，避免训练崩溃 (从 1e-3 调整到 3e-4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🔥 新增：梯度累积步数 (模拟更大的 batch size)
GRADIENT_ACCUMULATION_STEPS = 2

# 🔥 新增：混合精度训练 (节省显存，加速训练)
USE_AMP = True

# 🔥 新增：梯度裁剪阈值 (防止梯度爆炸)
GRADIENT_CLIP_VALUE = 1.0

# 数据路径 (修改为 TIF数据集)
TRAIN_IMG_DIR = './dataset/kaggle_3m/train'
VAL_IMG_DIR = './dataset/kaggle_3m/valid'
# TIF数据集不需要注释文件
# ===========================================

# --- 计算 Dice 和 IoU 的评估函数 ---
def calculate_metrics(pred, target, threshold=0.5):
    """计算 Dice 系数和 IoU"""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    return dice.item(), iou.item()

# ==========================================
# 🔥 新增：Focal Loss 处理类别不平衡
# ==========================================
class FocalLoss(nn.Module):
    """
    Focal Loss: 解决前景 - 背景类别不平衡问题
    gamma=2.0, alpha=0.75 是医学图像分割的推荐值
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # 🔥 使用 BCEWithLogitsLoss 替代 BCELoss 以支持混合精度训练
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# 定义联合损失函数 (Focal Loss + Dice，更强调难分样本)
def combined_loss(pred, target):
    # Focal Loss (更关注难分类的像素)
    focal = FocalLoss(alpha=0.75, gamma=2.0)(pred, target)
    
    # Dice Loss
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_loss = 1 - ((2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6))
    
    # 🔥 调整权重：Focal Loss 0.5 + Dice Loss 0.5
    return 0.5 * focal + 0.5 * dice_loss

# ==========================================
# 新增：混合精度训练 Scaler
# ==========================================
scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == 'cuda' else None

# ==========================================
# 新增：SwanLab 随机验证集可视化函数
# ==========================================
def log_predictions_to_swanlab(model, dataset, device, num_samples=4):
    """
    在训练结束后，随机抽取验证集样本进行可视化预测，并上传至 SwanLab
    """
    print(f"\n📸 正在生成 {num_samples} 张随机预测可视化图并上传 SwanLab...")
    model.eval() # 确保模型处于推理模式
    
    # 随机抽取样本索引
    indices = random.sample(range(len(dataset)), num_samples)
    swan_images = []
    
    # 定义反归一化参数 (还原为肉眼可看的 RGB 图像)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for idx in indices:
            image_tensor, mask_tensor = dataset[idx]
            
            # 送入显卡并增加 batch 维度
            input_tensor = image_tensor.unsqueeze(0).to(device)
            
            # 模型预测 (推理模式下只返回主输出 out_main)
            pred_tensor = model(input_tensor)
            pred_binary = (pred_tensor.squeeze().cpu() > 0.5).float().numpy()
            
            # 还原原始图像用于展示
            img_vis = image_tensor.cpu() * std + mean
            img_vis = torch.clamp(img_vis, 0, 1).permute(1, 2, 0).numpy()
            mask_vis = mask_tensor.squeeze().cpu().numpy()
            
            # --- 使用 Matplotlib 拼接对比图 ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_vis)
            axes[0].set_title(f"Original Image (Sample {idx})")
            axes[0].axis('off')
            
            axes[1].imshow(mask_vis, cmap='gray')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title("Predicted Mask (MK-UNet)")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 将 Matplotlib 图像转换为 SwanLab 格式
            swan_images.append(swanlab.Image(fig, caption=f"Validation Sample {idx}"))
            plt.close(fig)
            
    # 一次性上传所有对比图到 SwanLab
    swanlab.log({"Final_Visualization": swan_images})
    print("✅ 可视化结果已成功同步至 SwanLab 面板！")

# ==========================================
# 独立的训练核心函数 (已整合深度监督 Deep Supervision)
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path, scheduler=None, gradient_accumulation_steps=1, scaler=None):
    best_val_dice = 0.0
    patience = 10  # 🔥 增加早停耐心值到 10
    patience_counter = 0
    
    print("🏁 开始训练 Improved MK-UNet (开启深度监督 Deep Supervision)...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        train_dice_sum = 0
        train_iou_sum = 0
                
        optimizer.zero_grad()  # 🔥 在 batch 循环前清零梯度
                
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
                    
            # 🔥 使用混合精度训练
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # 🔥 1. 深度监督：接收主分支和两个辅助分支的预测结果
                outputs_main, outputs_up2, outputs_up3 = model(images)
                        
                # 🔥 2. 将辅助分支的特征图上采样 (放大) 到真实 Mask 的尺寸 (256x256)
                outputs_up2 = F.interpolate(outputs_up2, size=masks.shape[2:], mode='bilinear', align_corners=False)
                outputs_up3 = F.interpolate(outputs_up3, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        
                # 🔥 3. 计算多尺度联合损失 (权重分配：主输出 0.6, 辅助输出各 0.2)
                loss_main = criterion(outputs_main, masks)
                loss_up2 = criterion(outputs_up2, masks)
                loss_up3 = criterion(outputs_up3, masks)
                loss = 0.6 * loss_main + 0.2 * loss_up2 + 0.2 * loss_up3
                        
                # 🔥 梯度累积：缩放 loss 以模拟更大的 batch size
                loss = loss / gradient_accumulation_steps
            
            # 🔥 使用 scaler 进行反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                        
                # 🔥 梯度累积：每隔 accumulation_steps 步更新一次参数
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # 🔥 新增：梯度裁剪 (防止梯度爆炸)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
                    scaler.step(optimizer)
                    scaler.update()
                    # 🔥 OneCycleLR 需要在每个 step 后更新
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                        
                # 🔥 梯度累积：每隔 accumulation_steps 步更新一次参数
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # 🔥 新增：梯度裁剪 (防止梯度爆炸)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
                    optimizer.step()
                    # 🔥 OneCycleLR 需要在每个 step 后更新
                    scheduler.step()
                    optimizer.zero_grad()
                    
            train_loss += loss.item() * gradient_accumulation_steps  # 恢复原始 loss 用于日志
                    
            # 注意：计算准确率指标时，只看主分支的表现
            dice, iou = calculate_metrics(outputs_main, masks)
            train_dice_sum += dice
            train_iou_sum += iou
                    
            if (i + 1) % 10 == 0:
                print(f"   Batch {i+1}/{len(train_loader)} Total Loss: {loss.item()*gradient_accumulation_steps:.4f} (Main: {loss_main.item():.4f})")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice_sum / len(train_loader)
        avg_train_iou = train_iou_sum / len(train_loader)
        
        # --- 验证阶段 (使用混合精度) ---
        model.eval()
        val_loss = 0
        val_dice_sum = 0
        val_iou_sum = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                # 🔥 验证模式也使用混合精度
                with torch.amp.autocast('cuda', enabled=scaler is not None):
                    outputs_main = model(images)
                    loss = criterion(outputs_main, masks)
                
                val_loss += loss.item()
                dice, iou = calculate_metrics(outputs_main, masks)
                val_dice_sum += dice
                val_iou_sum += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)
        
        epoch_duration = time.time() - epoch_start
        
        # 打印详细日志
        print(f"\nEpoch [{epoch+1}/{num_epochs}] (耗时: {epoch_duration:.2f}s)")
        print(f"  Train - Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}")

        # 记录到 SwanLab
        swanlab.log({
            "Train/Loss": avg_train_loss,
            "Train/Dice": avg_train_dice,
            "Train/IoU": avg_train_iou,
            "Val/Loss": avg_val_loss,
            "Val/Dice": avg_val_dice,
            "Val/IoU": avg_val_iou,
            "Optimizer/Learning_Rate": optimizer.param_groups[0]['lr']  # OneCycleLR 不需要 get_last_lr()
        }, step=epoch+1)
        
        # 🔥 OneCycleLR 已经在每个 batch 后自动更新，不需要在这里调用 scheduler.step()

        # 早停与最佳模型保存策略
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型性能提升 (Dice: {avg_val_dice:.4f})，已保存至 {save_path}")
        else:
            patience_counter += 1
            print(f"⚠️ 验证集 Dice 未提升 (Patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"🛑 触发早停机制，训练在第 {epoch+1} 轮提前结束。")
                break

    total_duration = time.time() - start_time
    print(f"\n🎉 训练完全结束！总耗时: {total_duration/60:.2f} 分钟")


def main():
    print(f"🚀 使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 初始化 SwanLab 实验跟踪
    try:
        run = swanlab.init(
            project="Medical-Image-Segmentation-Graduation", 
            experiment_name="MKUNet-Stable-HighDice",
            config={
                "model": "MK-UNet (Stable Training + Gradient Clipping)",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(DEVICE),
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "gradient_clip_value": GRADIENT_CLIP_VALUE,
                "use_amp": USE_AMP,
                "lr_scheduler": "OneCycleLR with Warmup"
            }
        )
    except Exception as e:
        print(f"⚠️ SwanLab 云端连接失败：{e}\n   切换到本地离线模式...")
        run = swanlab.init(
            project="Medical-Image-Segmentation-Graduation", 
            experiment_name="MKUNet-Stable-HighDice-Local",
            config={
                "model": "MK-UNet (Stable Training + Gradient Clipping)",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(DEVICE),
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "gradient_clip_value": GRADIENT_CLIP_VALUE,
                "use_amp": USE_AMP,
                "lr_scheduler": "OneCycleLR with Warmup"
            },
            mode="local"
        )

    # 2. 准备数据集和加载器 (Transform已内置于dataset.py)
    print("\n📊 正在加载数据集 (自带 CLAHE 与同步几何增强)...")
    # TIF数据集不需要COCO
    # train_coco = COCO(TRAIN_ANN_FILE)
    # val_coco = COCO(VAL_ANN_FILE)

    train_dataset = TIFSegmentationDataset(TRAIN_IMG_DIR)
    val_dataset = TIFSegmentationDataset(VAL_IMG_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化模型
    print("🚀 正在加载创新版 Improved MK-UNet (带辅助输出头)...")
    model = ImprovedUNet(n_channels=3, n_classes=1).to(DEVICE)
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_model_mkunet.pth' 

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 当前 MK-UNet 参数量：{total_params/1e6:.4f} M\n")
    
    # 🔥 优化器：使用 AdamW (带权重衰减)，根据论文设置
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
    # 🔥 学习率调度器：使用带 Warmup 的 OneCycleLR (更稳定)
    # 🔥 前 10% 的 epoch 用于 warmup，然后使用余弦退火衰减
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,  # 初始 lr = max_lr/25
        final_div_factor=1000.0  # 最终 lr = max_lr/1000
    )
    
    # 🔥 混合精度训练 scaler
    scaler = torch.amp.GradScaler('cuda') if USE_AMP and DEVICE.type == 'cuda' else None
    
    print(f"\n🚀 开始训练 MK-UNet 模型...")
    print(f"   - 设备：{DEVICE}")
    print(f"   - 学习率：{LEARNING_RATE} (带 Warmup)")
    print(f"   - Batch Size: {BATCH_SIZE} (梯度累积：{GRADIENT_ACCUMULATION_STEPS})")
    print(f"   - 混合精度训练：{'启用' if scaler is not None else '禁用'}")
    print(f"   - 梯度裁剪：{'启用 (max_norm=1.0)' if GRADIENT_CLIP_VALUE > 0 else '禁用'}")
    print(f"   - 学习率调度器：OneCycleLR with Warmup")
    print("="*60)
    
    # 4. 调用训练核心函数 (传入 scaler)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combined_loss,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path=save_path,
        scheduler=scheduler,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        scaler=scaler
    )

    # ==========================================
    # 🔥 5. 训练结束：加载最佳权重并进行可视化检查
    # ==========================================
    print("\n🔍 正在加载最佳模型权重用于最终定性评估...")
    # 安全加载最佳权重
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    else:
        print("⚠️ 未找到最佳权重文件，将使用最后一代权重进行可视化。")
        
    # 调用可视化函数，从验证集中随机抽 4 张图
    log_predictions_to_swanlab(model, val_dataset, DEVICE, num_samples=4)

if __name__ == '__main__':
    main()