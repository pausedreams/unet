import torch
from predict import calculate_metrics
from dataset import TIFSegmentationDataset
from models.mkunet import ImprovedUNet
from torch.utils.data import DataLoader

device = torch.device('cuda')
model = ImprovedUNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load('checkpoints/best_model_mkunet_history.pth', map_location=device))
model.eval()

dataset = TIFSegmentationDataset('./dataset/kaggle_3m/test')
test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

print('='*60)
print('测试不同阈值对Dice的影响')
print('='*60)

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    dice_sum = 0
    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images.to(device))
            d, _ = calculate_metrics(outputs, masks.to(device), threshold=threshold)
            dice_sum += d
    
    avg_dice = dice_sum / len(test_loader)
    print(f'阈值 {threshold:.1f}: Test Dice = {avg_dice:.4f}')

print('='*60)
