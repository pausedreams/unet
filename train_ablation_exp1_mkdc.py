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
from dataset import TIFSegmentationDataset

# ================= 配置区域 =================
# 训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 60
LEARNING_RATE = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🔥 消融实验 Exp1: +MKDC (不使用深度监督、Focal Loss、AMP等)
GRADIENT_ACCUMULATION_STEPS = 2
USE_AMP = False  # ❌ 禁用 AMP
GRADIENT_CLIP_VALUE = 0.0  # ❌ 禁用梯度裁剪
USE_DEEP_SUPERVISION = False  # ❌ 禁用深度监督
USE_FOCAL_LOSS = False  # ❌ 禁用 Focal Loss

# 数据路径
TRAIN_IMG_DIR = './dataset/kaggle_3m/train'
VAL_IMG_DIR = './dataset/kaggle_3m/valid'
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

# --- 损失函数定义 ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, use_focal=False):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.use_focal = use_focal
        
        if use_focal:
            self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        
        if self.use_focal:
            focal_loss = self.focal(pred, target)
            return 0.4 * bce_loss + 0.4 * dice_loss + 0.2 * focal_loss
        
        return 0.5 * bce_loss + 0.5 * dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# --- 训练核心函数 ---
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None, gradient_accumulation_steps=1, grad_clip_value=0.0):
    model.train()
    train_loss = 0
    train_dice_sum = 0
    train_iou_sum = 0
    
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs_main, outputs_up2, outputs_up3 = model(images)
            
            loss_main = criterion(outputs_main, masks)
            if model.training and hasattr(model, 'outc_up2'):
                loss_up2 = criterion(outputs_up2, masks)
                loss_up3 = criterion(outputs_up3, masks)
                loss = 0.6 * loss_main + 0.2 * loss_up2 + 0.2 * loss_up3
            else:
                loss = loss_main
            
            loss = loss / gradient_accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if grad_clip_value > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                optimizer.step()
                optimizer.zero_grad()
        
        train_loss += loss.item() * gradient_accumulation_steps
        dice, iou = calculate_metrics(torch.sigmoid(outputs_main), masks)
        train_dice_sum += dice
        train_iou_sum += iou
        
        if (i + 1) % 10 == 0:
            print(f"   Batch {i+1}/{len(train_loader)} Total Loss: {loss.item() * gradient_accumulation_steps:.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice_sum / len(train_loader)
    avg_train_iou = train_iou_sum / len(train_loader)
    
    return avg_train_loss, avg_train_dice, avg_train_iou

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice_sum = 0
    val_iou_sum = 0
    
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs_main = model(images)
        loss = criterion(outputs_main, masks)
        
        val_loss += loss.item()
        dice, iou = calculate_metrics(torch.sigmoid(outputs_main), masks)
        val_dice_sum += dice
        val_iou_sum += iou
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice_sum / len(val_loader)
    avg_val_iou = val_iou_sum / len(val_loader)
    
    return avg_val_loss, avg_val_dice, avg_val_iou

def train_model(model, train_loader, val_loader, num_epochs, device, experiment_name="Ablation_Exp1_MKDC"):
    # 初始化 SwanLab
    try:
        run = swanlab.init(
            project="Medical-Image-Segmentation-Ablation",
            experiment_name=experiment_name,
            config={
                "model": "MK-UNet Ablation Exp1 (+MKDC)",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(device),
                "use_amp": USE_AMP,
                "gradient_clip": GRADIENT_CLIP_VALUE > 0,
                "deep_supervision": USE_DEEP_SUPERVISION,
                "focal_loss": USE_FOCAL_LOSS
            }
        )
    except Exception as e:
        print(f"⚠️ SwanLab 连接失败：{e}\n   切换到本地模式...")
        run = swanlab.init(
            project="Medical-Image-Segmentation-Ablation",
            experiment_name=experiment_name + "-Local",
            config={
                "model": "MK-UNet Ablation Exp1 (+MKDC)",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "device": str(device),
                "use_amp": USE_AMP,
                "gradient_clip": GRADIENT_CLIP_VALUE > 0,
                "deep_supervision": USE_DEEP_SUPERVISION,
                "focal_loss": USE_FOCAL_LOSS
            },
            mode="local"
        )
    
    model = model.to(device)
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/best_model_ablation_exp1.pth'
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 MK-UNet 参数量：{total_params/1e6:.4f} M\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    scaler = torch.amp.GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    
    criterion = CombinedLoss(use_focal=USE_FOCAL_LOSS)
    
    print(f"\n🚀 开始消融实验 Exp1: +MKDC")
    print(f"   - 设备：{device}")
    print(f"   - 学习率：{LEARNING_RATE} (带 Warmup)")
    print(f"   - Batch Size: {BATCH_SIZE} (梯度累积：{GRADIENT_ACCUMULATION_STEPS})")
    print(f"   - 混合精度训练：{'启用' if scaler is not None else '禁用'}")
    print(f"   - 梯度裁剪：{'启用' if GRADIENT_CLIP_VALUE > 0 else '禁用'}")
    print(f"   - 深度监督：{'启用' if USE_DEEP_SUPERVISION else '禁用'}")
    print(f"   - Focal Loss：{'启用' if USE_FOCAL_LOSS else '禁用'}")
    print("="*60)
    
    best_val_dice = 0
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        avg_train_loss, avg_train_dice, avg_train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            grad_clip_value=GRADIENT_CLIP_VALUE
        )
        
        avg_val_loss, avg_val_dice, avg_val_iou = validate(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] (耗时: {epoch_duration:.2f}s)")
        print(f"  Train - Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}")
        
        swanlab.log({
            "Train/Loss": avg_train_loss,
            "Train/Dice": avg_train_dice,
            "Train/IoU": avg_train_iou,
            "Val/Loss": avg_val_loss,
            "Val/Dice": avg_val_dice,
            "Val/IoU": avg_val_iou,
            "Optimizer/Learning_Rate": optimizer.param_groups[0]['lr']
        }, step=epoch+1)
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
            }, save_path)
            print(f"✅ 模型性能提升 (Dice: {best_val_dice:.4f})，已保存至 {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ 早停触发！最佳 Val Dice: {best_val_dice:.4f}")
                break
    
    print(f"\n🎉 训练完成！最佳 Val Dice: {best_val_dice:.4f}")
    swanlab.finish()

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    print("🚀 使用设备:", DEVICE)
    if DEVICE.type == 'cuda':
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n📊 正在加载数据集...")
    train_dataset = TIFSegmentationDataset(TRAIN_IMG_DIR, augment=True)
    val_dataset = TIFSegmentationDataset(VAL_IMG_DIR, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"   训练集样本数: {len(train_dataset)}")
    print(f"   验证集样本数: {len(val_dataset)}")
    
    print("\n🚀 正在加载 MK-UNet 模型（消融实验 Exp1: +MKDC）...")
    model = ImprovedUNet(n_channels=3, n_classes=1)
    
    train_model(model, train_loader, val_loader, NUM_EPOCHS, DEVICE, "Ablation_Exp1_MKDC")
