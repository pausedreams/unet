import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础工具函数与核心卷积模块
# ==========================================

# 通道洗牌：确保多路径特征充分融合
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

# MKDC 模块：多尺度特征提取核心 (丰富语义信息)
class MKDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1, 3, 5]):
        super(MKDC, self).__init__()
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                # 深度可分离卷积 (groups=in_channels)
                nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True)
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        # 并行计算并拼接
        out = torch.cat([dw(x) for dw in self.dwconvs], dim=1)
        # 通道洗牌，groups等于分支的数量
        return channel_shuffle(out, groups=len(self.dwconvs)) 

# MKIR 模块：多核倒置残差块
class MKIR(nn.Module):
    def __init__(self, in_c, out_c, expansion_factor=2, kernel_sizes=[1, 3, 5], use_mkdc=True):
        super(MKIR, self).__init__()
        ex_c = in_c * expansion_factor
        
        # 1. 升维 (1x1 Conv)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, bias=False), 
            nn.BatchNorm2d(ex_c), 
            nn.ReLU6(inplace=True)
        )
        
        # 2. 多核深度卷积 (MKDC) 或 标准卷积
        self.use_mkdc = use_mkdc
        if use_mkdc:
            self.feature_extractor = MKDC(ex_c, kernel_sizes=kernel_sizes)
            mkdc_out_c = ex_c * len(kernel_sizes)
        else:
            # 消融模式：用标准 3x3 深度卷积替代多核并行
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(ex_c, ex_c, 3, padding=1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True)
            )
            mkdc_out_c = ex_c
        
        # 3. 降维 (1x1 Conv)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(mkdc_out_c, out_c, 1, bias=False), 
            nn.BatchNorm2d(out_c)
        )
        
        # 4. 残差连接 (匹配维度)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.pconv2(self.feature_extractor(self.pconv1(x))) + self.skip(x)

# ==========================================
# 2. 注意力机制模块 (Attention Blocks)
# ==========================================

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(1, in_planes // ratio)
        
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

# MKIRA: 多核倒置残差注意力模块 
class MKIRA(nn.Module):
    def __init__(self, in_c, out_c, use_attention=True, use_mkdc=True):
        super(MKIRA, self).__init__()
        self.use_attention = use_attention
        if use_attention:
            self.ca = ChannelAttention(in_c)
            self.sa = SpatialAttention()
        self.mkir = MKIR(in_c, out_c, use_mkdc=use_mkdc) 

    def forward(self, x):
        if self.use_attention:
            x = self.ca(x) * x  # 通道加权
            x = self.sa(x) * x  # 空间加权
        return self.mkir(x)     # 特征提取与降维

# 导师修改版 GAG: 标准注意力门 (Attention Gate)
# 去除了容易报错的 groups=2，改用 1x1 卷积，更轻量且更符合学术标准
class GAG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GAG, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 返回被注意力门过滤后的跳跃连接特征
        return x * psi 

# ==========================================
# 3. MK-UNet 主类
# ==========================================

# 改进后的下采样块
class ImprovedDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_mkdc=True, use_attention=True):
        super(ImprovedDownBlock, self).__init__()
        self.mkir = MKIR(in_channels, out_channels, use_mkdc=use_mkdc)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.mkir(x)
        skip = x
        x = self.maxpool(x)
        return x, skip

# 改进后的上采样块
class ImprovedUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_gag=True, use_attention=True, use_mkdc=True):
        super(ImprovedUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.use_gag = use_gag
        if use_gag:
            # 修正的 GAG 模块
            self.gag = GAG(F_g=in_channels, F_l=out_channels, F_int=out_channels // 2)
        else:
            self.gag = nn.Identity()  # 消融模式：直接跳过 GAG
        # 拼接后通道数为 in_channels + out_channels
        self.mkira = MKIRA(in_channels + out_channels, out_channels, 
                          use_attention=use_attention, use_mkdc=use_mkdc)

    def forward(self, x, skip):
        x = self.up(x)
        # 用解码层特征 (g=x) 去过滤编码层特征 (x=skip)
        if self.use_gag:
            skip = self.gag(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        x = self.mkira(x)
        return x

# 最终创新模型类: Improved MK-UNet
class ImprovedUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, 
                 use_mkdc=True, use_attention=True, use_gag=True, use_deep_sup=True):
        super(ImprovedUNet, self).__init__()
        self.use_deep_sup = use_deep_sup
        filters = [16, 32, 64, 96, 160]

        # 编码器与瓶颈层
        self.down1 = ImprovedDownBlock(n_channels, filters[0], use_mkdc=use_mkdc, use_attention=use_attention)
        self.down2 = ImprovedDownBlock(filters[0], filters[1], use_mkdc=use_mkdc, use_attention=use_attention)
        self.down3 = ImprovedDownBlock(filters[1], filters[2], use_mkdc=use_mkdc, use_attention=use_attention)
        self.down4 = ImprovedDownBlock(filters[2], filters[3], use_mkdc=use_mkdc, use_attention=use_attention)
        self.bottleneck = MKIR(filters[3], filters[4], use_mkdc=use_mkdc)

        # 解码器
        self.up1 = ImprovedUpBlock(filters[4], filters[3], use_gag=use_gag, use_attention=use_attention, use_mkdc=use_mkdc)
        self.up2 = ImprovedUpBlock(filters[3], filters[2], use_gag=use_gag, use_attention=use_attention, use_mkdc=use_mkdc)
        self.up3 = ImprovedUpBlock(filters[2], filters[1], use_gag=use_gag, use_attention=use_attention, use_mkdc=use_mkdc)
        self.up4 = ImprovedUpBlock(filters[1], filters[0], use_gag=use_gag, use_attention=use_attention, use_mkdc=use_mkdc)

        # 深度监督核心：新增辅助分类头 (Auxiliary Classifiers)
        self.outc_main = nn.Conv2d(filters[0], n_classes, 1)  # 最终主输出
        if use_deep_sup:
            self.outc_up2 = nn.Conv2d(filters[2], n_classes, 1)  # 中间层2输出
            self.outc_up3 = nn.Conv2d(filters[1], n_classes, 1)  # 中间层3输出
        else:
            # 消融模式：用 Identity 占位，避免 forward 中引用不存在的属性
            self.outc_up2 = nn.Identity()
            self.outc_up3 = nn.Identity()

    def forward(self, x):
        # 编码与瓶颈
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        x5 = self.bottleneck(x4)

        # 解码与辅助特征提取
        x_up1 = self.up1(x5, skip4)
        x_up2 = self.up2(x_up1, skip3)
        x_up3 = self.up3(x_up2, skip2)
        x_up4 = self.up4(x_up3, skip1)

        # 最终的主输出
        out_main = torch.sigmoid(self.outc_main(x_up4))

        # 如果是训练模式且启用了深度监督，同时返回中间层的预测结果
        if self.training and self.use_deep_sup:
            out_up2 = torch.sigmoid(self.outc_up2(x_up2))
            out_up3 = torch.sigmoid(self.outc_up3(x_up3))
            return out_main, out_up2, out_up3
        
        # 评估/推理模式，只返回最终最高精度的输出
        return out_main


# ==========================================
# 4. 消融实验模型工厂函数
# ==========================================

# 消融变体配置表
ABLATION_CONFIGS = {
    'full': {
        'use_mkdc': True, 'use_attention': True, 'use_gag': True, 'use_deep_sup': True,
        'name': 'Full Model (Ours)'
    },
    'w/o_mkdc': {
        'use_mkdc': False, 'use_attention': True, 'use_gag': True, 'use_deep_sup': True,
        'name': 'w/o MKDC'
    },
    'w/o_attention': {
        'use_mkdc': True, 'use_attention': False, 'use_gag': True, 'use_deep_sup': True,
        'name': 'w/o Attention'
    },
    'w/o_gag': {
        'use_mkdc': True, 'use_attention': True, 'use_gag': False, 'use_deep_sup': True,
        'name': 'w/o GAG'
    },
    'w/o_deep_sup': {
        'use_mkdc': True, 'use_attention': True, 'use_gag': True, 'use_deep_sup': False,
        'name': 'w/o Deep Supervision'
    },
}

def create_ablation_model(variant='full', n_channels=3, n_classes=1):
    """
    根据消融实验变体名称创建模型
    
    Args:
        variant: 'full', 'w/o_mkdc', 'w/o_attention', 'w/o_gag', 'w/o_deep_sup'
        n_channels: 输入通道数
        n_classes: 输出类别数
    
    Returns:
        配置好的 ImprovedUNet 实例
    """
    if variant not in ABLATION_CONFIGS:
        raise ValueError(f"未知变体: {variant}. 可选: {list(ABLATION_CONFIGS.keys())}")
    
    cfg = ABLATION_CONFIGS[variant].copy()
    cfg.pop('name')  # 移除名称字段
    return ImprovedUNet(n_channels=n_channels, n_classes=n_classes, **cfg)


if __name__ == '__main__':
    # 测试所有消融变体
    print("=" * 60)
    print("消融变体模型测试")
    print("=" * 60)
    
    for variant, cfg in ABLATION_CONFIGS.items():
        try:
            model = create_ablation_model(variant)
            x = torch.randn(2, 3, 256, 256)
            
            # 推理模式测试
            model.eval()
            with torch.no_grad():
                y = model(x)
            
            # 训练模式测试（深度监督变体应返回3个输出）
            model.train()
            outputs = model(x)
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n✅ {cfg['name']}")
            print(f"   输入: {x.shape}")
            print(f"   输出: {y.shape}")
            print(f"   参数量: {total_params/1e6:.4f}M")
            
        except Exception as e:
            print(f"\n❌ {cfg['name']} 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("所有变体测试完成！")
    print("=" * 60)
