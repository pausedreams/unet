# 🚀 消融实验快速开始指南

## 📋 前置条件

确保你已经：
- ✅ 安装了所有依赖 (`pip install -r requirements.txt`)
- ✅ 准备好了数据集（`dataset/kaggle_3m/`）
- ✅ 配置了 Python 环境（Python 3.8+，PyTorch 1.9+）

---

## 🎯 方案选择

### 方案 A：运行单个实验（推荐新手）

适合想逐步了解每个模块作用的用户。

#### Step 1: 运行 Baseline UNet（已有数据可跳过）

```bash
python train_unet.py
```

**预期结果：**
- Val Dice: ~0.75
- 训练时间: ~90 分钟
- 模型保存: `checkpoints/best_model_unet.pth`

#### Step 2: 运行 Exp1: +MKDC

```bash
python train_ablation_exp1_mkdc.py
```

**预期结果：**
- Val Dice: ~0.78 (+0.03)
- 训练时间: ~90 分钟
- 模型保存: `checkpoints/best_model_ablation_exp1.pth`

**对比分析：**
```python
# 比较两个模型的参数量
Baseline UNet: 34.5M 参数
+MKDC:         0.8M 参数  (减少 97.7%!)

# 性能对比
Baseline UNet: Dice 0.75
+MKDC:         Dice 0.78  (提升 4%)
```

---

### 方案 B：自动运行所有实验（推荐进阶用户）

适合想快速获取完整消融实验结果的用户。

#### Step 1: 启动自动化脚本

```bash
python run_ablation_study.py
```

脚本会询问是否开始，输入 `y` 确认。

**执行流程：**
1. 自动运行 Exp 0 (Baseline UNet)
2. 等待 5 秒
3. 自动运行 Exp 1 (+MKDC)
4. 生成 JSON 报告
5. 显示总结表格

**预计总耗时：** 约 180 分钟（2 个实验 × 90 分钟）

#### Step 2: 查看实验报告

实验完成后会自动生成报告文件：
```
ablation_study_report_YYYYMMDD_HHMMSS.json
```

---

### 方案 C：仅可视化已有结果

如果你已经有实验数据（从 SwanLab 或之前的训练），可以直接可视化。

```bash
python visualize_ablation_results.py
```

**生成的图表：**
- 📊 `ablation_bar_chart.png` - 消融实验对比柱状图
- 📊 `module_contribution.png` - 模块贡献度分析图
- 📊 `training_efficiency.png` - 训练效率对比图

---

## 📊 监控实验进度

### 方法 1：终端实时输出

训练过程中会实时显示：
```
Epoch [1/60] (耗时: 96.78s)
  Train - Loss: 0.6209 | Dice: 0.0467 | IoU: 0.0246
  Val   - Loss: 0.6205 | Dice: 0.0625 | IoU: 0.0357
✅ 模型性能提升 (Dice: 0.0625)，已保存至 checkpoints/...
```

### 方法 2：SwanLab 云端监控

访问 SwanLab 查看实时曲线：
- 项目页面：https://swanlab.cn/@pausedreams/Medical-Image-Segmentation-Ablation
- 实验名称：`Ablation_Exp1_MKDC`

**关键指标：**
- `Train/Loss` - 训练损失
- `Val/Dice` - 验证集 Dice（核心指标）
- `Optimizer/Learning_Rate` - 学习率变化

---

## 🔍 结果分析

### 查看实验报告

```bash
# 查看最新的 JSON 报告
cat ablation_study_report_*.json
```

**报告内容示例：**
```json
{
  "timestamp": "2026-03-21 21:30:00",
  "total_experiments": 2,
  "completed": 2,
  "failed": 0,
  "experiments": [
    {
      "id": "exp0",
      "name": "Baseline_UNet",
      "status": "success",
      "duration_minutes": 89.5,
      "expected_dice": 0.75
    },
    {
      "id": "exp1",
      "name": "MKDC_Only",
      "status": "success",
      "duration_minutes": 91.2,
      "expected_dice": 0.78
    }
  ]
}
```

### 生成可视化图表

```bash
python visualize_ablation_results.py
```

打开生成的 PNG 图片查看对比结果。

---

## 💡 常见问题

### Q1: 训练中途被中断怎么办？

**A:** 模型会自动保存最佳权重到 `checkpoints/`，可以从中断的 epoch 继续训练。

### Q2: 显存不足怎么办？

**A:** 减小 batch size：
```python
# 在训练脚本中修改
BATCH_SIZE = 2  # 从 4 降低到 2
```

### Q3: 如何加快实验速度？

**A:** 
1. 减少训练轮数：
```python
NUM_EPOCHS = 30  # 从 60 降低到 30
```

2. 使用更小的数据集进行测试：
```python
# 只使用前 100 个样本
train_dataset = TIFSegmentationDataset(TRAIN_IMG_DIR, augment=True, max_samples=100)
```

### Q4: 如何添加新的消融实验？

**A:** 
1. 复制 `train_ablation_exp1_mkdc.py`
2. 修改配置（启用/禁用特定模块）
3. 更新 `run_ablation_study.py` 中的 EXPERIMENTS 列表
4. 运行自动化脚本

**示例：创建 Exp2 (+GAG)**
```python
# train_ablation_exp2_gag.py
USE_DEEP_SUPERVISION = False  # ❌ 禁用
USE_FOCAL_LOSS = False        # ❌ 禁用
# ... 其他配置与 Exp1 相同
```

---

## 📈 下一步行动

### 短期目标（今天）
- ✅ 运行 Exp1 (+MKDC) 实验
- ✅ 对比 Baseline 和 +MKDC 的性能差异
- ✅ 查看 SwanLab 上的训练曲线

### 中期目标（本周）
- ⏳ 完成所有 7 个消融实验
- ⏳ 生成完整的对比报告
- ⏳ 分析各模块的贡献度

### 长期目标（本月）
- 🎯 撰写消融实验分析报告
- 🎯 准备论文/毕业设计的实验部分
- 🎯 上传最终结果到 GitHub

---

## 🤝 获取帮助

遇到问题？
- 📖 查看 [ABLATION_STUDY.md](ABLATION_STUDY.md) 详细文档
- 🐛 提交 GitHub Issue
- 💬 联系项目维护者

---

## 🎉 成功标志

当你看到以下输出时，说明消融实验成功完成：

```
🎉 所有实验已完成！
📊 成功率: 2/2 (100.0%)

================================================================================
📊 消融实验总结报告
================================================================================

实验ID     实验名称                  状态       耗时(min)    预期Dice    
--------------------------------------------------------------------------------
exp0       Baseline_UNet            ✅ 成功     89.50       0.75      
exp1       MKDC_Only                ✅ 成功     91.20       0.78      
--------------------------------------------------------------------------------

📄 详细报告已保存至: ablation_study_report_20260321_213000.json
```

**恭喜！你已成功完成消融实验！** 🎊
