import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        """
        学术级医学图像数据集加载器
        :param coco: pycocotools 初始化的 COCO 对象
        :param image_dir: 图像所在文件夹路径
        :param transform: (遗留参数，内部已采用更安全的同步变换替代)
        """
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        
        # 💡 导师黑科技：智能判断当前是否为训练集 (无需修改外部训练脚本)
        # 只有训练集才会开启随机翻转和旋转，验证集/测试集只做 Resize 和 CLAHE
        self.is_train = 'train' in image_dir.lower()

    def __len__(self):
        return len(self.image_ids)

    def apply_clahe(self, image_np):
        """
        像素级抗噪增强：对比度受限自适应直方图均衡化 (CLAHE)
        有效解决医学图像边缘模糊、对比度低的问题
        """
        # 创建 CLAHE 对象 (clipLimit 控制对比度阈值，tileGridSize 决定局部窗口大小)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 将 RGB 图像转换到 LAB 颜色空间，仅对亮度通道(L)进行增强
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return image_clahe
        elif len(image_np.shape) == 2 or image_np.shape[2] == 1:
            return clahe.apply(image_np)
        return image_np

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # 1. 加载图像并进行 CLAHE 增强
        image_np = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
        image_np = self.apply_clahe(image_np)  # 注入像素级抗噪灵魂
        image_pil = Image.fromarray(image_np)

        # 2. 创建精准的二值化掩码 (Mask)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            mask_np = np.maximum(mask_np, self.coco.annToMask(ann))
        mask_pil = Image.fromarray(mask_np)

        # ==========================================
        # 3. 同步空间变换 (Synchronized Spatial Transforms)
        # ==========================================
        
        # 首先，统一将 Image 和 Mask Resize 到 256x256
        # Image 用双线性插值(BILINEAR)，Mask 必须用最近邻(NEAREST)防止产生0.5这种非法类别值
        image_pil = TF.resize(image_pil, (256, 256), interpolation=Image.BILINEAR)
        mask_pil = TF.resize(mask_pil, (256, 256), interpolation=Image.NEAREST)

        # 如果是训练集，则进行严格对齐的几何数据增强
        if self.is_train:
            # 50% 概率水平翻转
            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)
                    
            # 50% 概率垂直翻转
            if random.random() > 0.5:
                image_pil = TF.vflip(image_pil)
                mask_pil = TF.vflip(mask_pil)
                    
            # 50% 概率随机旋转 (-20度 到 20度，扩大角度范围)
            if random.random() > 0.5:
                angle = random.randint(-20, 20)
                image_pil = TF.rotate(image_pil, angle, interpolation=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)
                    
            # 🔥 新增：随机缩放 (85% - 115%，更激进的范围)
            if random.random() > 0.5:
                scale = random.uniform(0.85, 1.15)
                new_size = int(256 * scale)
                # 确保 new_size 至少为 256
                if new_size >= 256:
                    image_pil = TF.resize(image_pil, (new_size, new_size), interpolation=Image.BILINEAR)
                    mask_pil = TF.resize(mask_pil, (new_size, new_size), interpolation=Image.NEAREST)
                    # 随机裁剪回 256x256
                    i, j = random.randint(0, new_size - 256), random.randint(0, new_size - 256)
                    image_pil = TF.crop(image_pil, i, j, 256, 256)
                    mask_pil = TF.crop(mask_pil, i, j, 256, 256)
                    
            # 🔥 新增：颜色抖动 (更强的增强)
            if random.random() > 0.5:
                image_pil = TF.adjust_brightness(image_pil, random.uniform(0.8, 1.2))
                image_pil = TF.adjust_contrast(image_pil, random.uniform(0.8, 1.2))
                image_pil = TF.adjust_saturation(image_pil, random.uniform(0.9, 1.1))

        # ==========================================
        # 4. 张量转换与标准化
        # ==========================================
        
        # 将 PIL 转换为 Tensor (会自动将像素值缩放到 0~1 之间)
        image_tensor = TF.to_tensor(image_pil)
        
        # Mask 转换为张量，并增加通道维度 [1, H, W]
        mask_tensor = torch.as_tensor(np.array(mask_pil), dtype=torch.float32).unsqueeze(0)

        # 对 Image 进行标准化 (医疗图像推荐使用 ImageNet 均值方差作为预训练基础)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image_tensor, mask_tensor


class TIFSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        数据集加载器 for TIF格式的LGG Segmentation Dataset
        :param root_dir: 数据集根目录，包含多个患者文件夹
        :param transform: (遗留参数)
        """
        self.root_dir = root_dir
        self.image_mask_pairs = []
        
        # 遍历所有患者文件夹
        for patient_folder in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_folder)
            if os.path.isdir(patient_path):
                # 找到所有图像文件（不包含_mask）
                for file in os.listdir(patient_path):
                    if file.endswith('.tif') and not file.endswith('_mask.tif'):
                        image_path = os.path.join(patient_path, file)
                        mask_path = os.path.join(patient_path, file.replace('.tif', '_mask.tif'))
                        if os.path.exists(mask_path):
                            self.image_mask_pairs.append((image_path, mask_path))
        
        # 判断是否为训练集
        self.is_train = 'train' in root_dir.lower()

    def __len__(self):
        return len(self.image_mask_pairs)

    def apply_clahe(self, image_np):
        """
        像素级抗噪增强：对比度受限自适应直方图均衡化 (CLAHE)
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return image_clahe
        elif len(image_np.shape) == 2 or image_np.shape[2] == 1:
            return clahe.apply(image_np)
        return image_np

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        
        # 加载图像和掩码
        try:
            image_pil = Image.open(image_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('L')  # 灰度掩码
        except Exception as e:
            print(f"Error loading {image_path} or {mask_path}: {e}")
            # 返回一个空的掩码
            image_pil = Image.new('RGB', (256, 256), (0, 0, 0))
            mask_pil = Image.new('L', (256, 256), 0)
        
        # 暂时移除CLAHE以避免问题
        # image_np = np.array(image_pil)
        # image_np = self.apply_clahe(image_np)
        # image_pil = Image.fromarray(image_np)
        
        # 确保尺寸为256x256（如果不是的话）
        if image_pil.size != (256, 256):
            image_pil = TF.resize(image_pil, (256, 256), interpolation=Image.BILINEAR)
            mask_pil = TF.resize(mask_pil, (256, 256), interpolation=Image.NEAREST)
        
        # 数据增强（仅训练集）
        if self.is_train:
            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)
            
            if random.random() > 0.5:
                image_pil = TF.vflip(image_pil)
                mask_pil = TF.vflip(mask_pil)
            
            if random.random() > 0.5:
                angle = random.randint(-15, 15)  # 🔥 从 -20~20 降低到 -15~15
                image_pil = TF.rotate(image_pil, angle, interpolation=Image.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)
            
            # 🔥 优化：随机缩放 (缩小范围，提高稳定性)
            if random.random() > 0.5:
                scale = random.uniform(0.95, 1.05)  # 🔥 从 90%-110% 缩小到 95%-105%
                new_size = int(256 * scale)
                # 确保 new_size 至少为 256
                if new_size >= 256:
                    image_pil = TF.resize(image_pil, (new_size, new_size), interpolation=Image.BILINEAR)
                    mask_pil = TF.resize(mask_pil, (new_size, new_size), interpolation=Image.NEAREST)
                    # 随机裁剪回 256x256
                    i, j = random.randint(0, new_size - 256), random.randint(0, new_size - 256)
                    image_pil = TF.crop(image_pil, i, j, 256, 256)
                    mask_pil = TF.crop(mask_pil, i, j, 256, 256)
            
            # 🔥 优化：颜色抖动 (降低强度，避免图像失真)
            if random.random() > 0.5:
                brightness = random.uniform(0.95, 1.05)  # 🔥 降低亮度变化范围
                contrast = random.uniform(0.95, 1.05)    # 🔥 降低对比度变化范围
                saturation = random.uniform(0.95, 1.05)  # 🔥 降低饱和度变化范围
                image_pil = TF.adjust_brightness(image_pil, brightness)
                image_pil = TF.adjust_contrast(image_pil, contrast)
                image_pil = TF.adjust_saturation(image_pil, saturation)
        
        # 转换为张量
        image_tensor = TF.to_tensor(image_pil)
        mask_tensor = torch.as_tensor(np.array(mask_pil), dtype=torch.float32).unsqueeze(0) / 255.0  # 归一化到0-1
        
        # 标准化
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image_tensor, mask_tensor