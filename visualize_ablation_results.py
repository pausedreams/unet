"""
📊 消融实验结果可视化脚本

从 SwanLab 或 JSON 报告中读取实验数据，生成对比图表。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_data(report_file=None):
    """加载实验数据"""
    if report_file and os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 默认示例数据（实际使用时应从 SwanLab 或训练日志中提取）
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [
            {"id": "exp0", "name": "Baseline UNet", "status": "success", "expected_dice": 0.75, "actual_dice": 0.74},
            {"id": "exp1", "name": "+MKDC", "status": "success", "expected_dice": 0.78, "actual_dice": 0.77},
            {"id": "exp2", "name": "+GAG", "status": "success", "expected_dice": 0.80, "actual_dice": 0.79},
            {"id": "exp3", "name": "+DeepSup", "status": "success", "expected_dice": 0.82, "actual_dice": 0.81},
            {"id": "exp4", "name": "+Focal Loss", "status": "success", "expected_dice": 0.83, "actual_dice": 0.82},
            {"id": "exp5", "name": "+AMP+GradClip", "status": "success", "expected_dice": 0.84, "actual_dice": 0.83},
            {"id": "exp6", "name": "Full Model", "status": "success", "expected_dice": 0.85, "actual_dice": 0.85}
        ]
    }

def plot_ablation_bar_chart(data, save_path="ablation_results.png"):
    """绘制消融实验柱状图"""
    experiments = data["experiments"]
    
    names = [exp["name"] for exp in experiments]
    expected_dice = [exp["expected_dice"] for exp in experiments]
    actual_dice = [exp.get("actual_dice", exp["expected_dice"]) for exp in experiments]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width/2, expected_dice, width, label='预期 Dice', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, actual_dice, width, label='实际 Dice', color='#4CAF50', alpha=0.8)
    
    ax.set_xlabel('实验配置', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Improved MK-UNet 消融实验结果对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱状图上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 添加基线参考线
    baseline_dice = expected_dice[0]
    ax.axhline(y=baseline_dice, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'基线 ({baseline_dice:.2f})')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 柱状图已保存至: {save_path}")
    plt.show()

def plot_module_contribution(data, save_path="module_contribution.png"):
    """绘制模块贡献度分析图"""
    experiments = data["experiments"]
    
    # 计算每个模块的贡献
    baseline = experiments[0]["expected_dice"]
    modules = []
    contributions = []
    
    module_names = [
        "Baseline",
        "MKDC",
        "GAG", 
        "DeepSup",
        "Focal Loss",
        "AMP+GC",
        "OneCycleLR"
    ]
    
    for i, exp in enumerate(experiments):
        if i == 0:
            modules.append(module_names[i])
            contributions.append(exp["expected_dice"])
        else:
            prev_dice = experiments[i-1]["expected_dice"]
            curr_dice = exp["expected_dice"]
            contribution = curr_dice - prev_dice
            
            modules.append(f"+{module_names[i]}")
            contributions.append(contribution)
    
    colors = ['#FF5722' if i == 0 else '#4CAF50' if c > 0 else '#F44336' 
              for i, c in enumerate(contributions)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(modules))
    bars = ax.bar(x, contributions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('模块添加顺序', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice 提升幅度', fontsize=12, fontweight='bold')
    ax.set_title('各模块对模型性能的贡献度分析', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 添加数值标签
    for bar, contrib in zip(bars, contributions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{contrib:+.3f}' if contrib != 0 else f'{contrib:.3f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 贡献度分析图已保存至: {save_path}")
    plt.show()

def plot_training_efficiency(data, save_path="training_efficiency.png"):
    """绘制训练效率对比图"""
    experiments = data["experiments"]
    
    names = [exp["name"] for exp in experiments]
    durations = [exp.get("duration_minutes", 90) for exp in experiments]
    dice_scores = [exp.get("actual_dice", exp["expected_dice"]) for exp in experiments]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：训练时间
    colors_time = ['#FF9800' if d > 100 else '#4CAF50' if d < 85 else '#FFC107' for d in durations]
    bars1 = ax1.bar(range(len(names)), durations, color=colors_time, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('实验配置', fontsize=12, fontweight='bold')
    ax1.set_ylabel('训练时间 (分钟)', fontsize=12, fontweight='bold')
    ax1.set_title('各实验训练耗时对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, dur in zip(bars1, durations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{dur:.1f}min',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 右图：性能-时间权衡
    scatter = ax2.scatter(durations, dice_scores, s=200, c=dice_scores, 
                         cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=2)
    ax2.set_xlabel('训练时间 (分钟)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Val Dice', fontsize=12, fontweight='bold')
    ax2.set_title('性能-时间权衡分析', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加实验标签
    for i, (name, dur, dice) in enumerate(zip(names, durations, dice_scores)):
        ax2.annotate(name.split()[0], (dur, dice), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, ax=ax2, label='Dice Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练效率图已保存至: {save_path}")
    plt.show()

def generate_summary_table(data):
    """生成总结表格"""
    print("\n" + "="*100)
    print("📊 消融实验结果汇总表")
    print("="*100)
    
    print(f"\n{'实验ID':<10} {'实验名称':<20} {'预期Dice':<12} {'实际Dice':<12} {'差异':<10} {'状态':<10}")
    print("-"*100)
    
    for exp in data["experiments"]:
        expected = exp["expected_dice"]
        actual = exp.get("actual_dice", expected)
        diff = actual - expected
        status = "✅" if exp["status"] == "success" else "❌"
        
        print(f"{exp['id']:<10} {exp['name']:<20} {expected:<12.4f} {actual:<12.4f} {diff:+.4f}    {status:<10}")
    
    print("="*100)
    
    # 计算总体统计
    successful = [exp for exp in data["experiments"] if exp["status"] == "success"]
    if successful:
        avg_expected = np.mean([exp["expected_dice"] for exp in successful])
        avg_actual = np.mean([exp.get("actual_dice", exp["expected_dice"]) for exp in successful])
        max_improvement = max([exp.get("actual_dice", exp["expected_dice"]) for exp in successful]) - successful[0]["expected_dice"]
        
        print(f"\n📈 统计摘要:")
        print(f"   • 成功实验数: {len(successful)}/{len(data['experiments'])}")
        print(f"   • 平均预期 Dice: {avg_expected:.4f}")
        print(f"   • 平均实际 Dice: {avg_actual:.4f}")
        print(f"   • 最大性能提升: +{max_improvement:.4f}")
        print(f"   • 最佳模型: {max(successful, key=lambda x: x.get('actual_dice', x['expected_dice']))['name']}")

def main():
    """主函数"""
    print("📊 Improved MK-UNet 消融实验结果可视化")
    print("="*80)
    
    # 查找最新的报告文件
    report_files = [f for f in os.listdir('.') if f.startswith('ablation_study_report_') and f.endswith('.json')]
    
    if report_files:
        latest_report = sorted(report_files)[-1]
        print(f"\n📄 使用最新报告: {latest_report}")
        data = load_experiment_data(latest_report)
    else:
        print("\n⚠️  未找到实验报告，使用示例数据...")
        data = load_experiment_data()
    
    # 生成可视化
    print("\n🎨 正在生成可视化图表...")
    
    try:
        plot_ablation_bar_chart(data, save_path="ablation_bar_chart.png")
        plot_module_contribution(data, save_path="module_contribution.png")
        plot_training_efficiency(data, save_path="training_efficiency.png")
        generate_summary_table(data)
        
        print("\n✅ 所有可视化完成！")
        print("\n生成的文件:")
        print("  📊 ablation_bar_chart.png - 消融实验对比柱状图")
        print("  📊 module_contribution.png - 模块贡献度分析图")
        print("  📊 training_efficiency.png - 训练效率对比图")
        
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
