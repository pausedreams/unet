"""
🔬 消融实验自动化执行脚本

此脚本会依次运行所有消融实验，并生成对比报告。
"""

import subprocess
import time
import json
import os
from datetime import datetime

# 实验配置
EXPERIMENTS = [
    {
        "id": "exp0",
        "name": "Baseline_UNet",
        "script": "train_unet.py",
        "description": "标准 UNet 基线模型",
        "expected_dice": 0.75
    },
    {
        "id": "exp1", 
        "name": "MKDC_Only",
        "script": "train_ablation_exp1_mkdc.py",
        "description": "添加 MKDC 多核卷积模块",
        "expected_dice": 0.78
    },
    # 后续实验可以逐步添加
]

def run_experiment(exp_config):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"🚀 开始实验: {exp_config['name']}")
    print(f"📝 描述: {exp_config['description']}")
    print(f"🎯 预期 Dice: {exp_config['expected_dice']}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # 运行训练脚本
        result = subprocess.run(
            ["python", exp_config["script"]],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ 实验 {exp_config['name']} 完成！")
            print(f"⏱️  耗时: {duration/60:.2f} 分钟")
            return {
                "status": "success",
                "duration_minutes": duration/60,
                "experiment": exp_config
            }
        else:
            print(f"\n❌ 实验 {exp_config['name']} 失败！")
            return {
                "status": "failed",
                "duration_minutes": duration/60,
                "experiment": exp_config
            }
    
    except Exception as e:
        print(f"\n❌ 实验 {exp_config['name']} 异常: {e}")
        return {
            "status": "error",
            "error": str(e),
            "experiment": exp_config
        }

def generate_report(results):
    """生成实验报告"""
    print("\n\n" + "="*80)
    print("📊 消融实验总结报告")
    print("="*80 + "\n")
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "completed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "experiments": []
    }
    
    print(f"📅 实验时间: {report['timestamp']}")
    print(f"✅ 完成实验: {report['completed']}/{report['total_experiments']}")
    print(f"❌ 失败实验: {report['failed']}/{report['total_experiments']}\n")
    
    print("-"*80)
    print(f"{'实验ID':<10} {'实验名称':<25} {'状态':<10} {'耗时(min)':<12} {'预期Dice':<10}")
    print("-"*80)
    
    for result in results:
        exp = result["experiment"]
        status = "✅ 成功" if result["status"] == "success" else "❌ 失败"
        duration = f"{result.get('duration_minutes', 0):.2f}"
        
        print(f"{exp['id']:<10} {exp['name']:<25} {status:<10} {duration:<12} {exp['expected_dice']:<10}")
        
        report["experiments"].append({
            "id": exp["id"],
            "name": exp["name"],
            "status": result["status"],
            "duration_minutes": result.get("duration_minutes", 0),
            "expected_dice": exp["expected_dice"],
            "description": exp["description"]
        })
    
    print("-"*80)
    
    # 保存报告
    report_file = f"ablation_study_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细报告已保存至: {report_file}")
    
    return report

def main():
    """主函数"""
    print("🔬 Improved MK-UNet 消融实验自动化系统")
    print("="*80)
    print(f"📋 计划运行 {len(EXPERIMENTS)} 个实验")
    print(f"⏱️  预计总耗时: 约 {len(EXPERIMENTS) * 90} 分钟")
    print("="*80)
    
    # 确认开始
    response = input("\n是否开始运行消融实验？(y/n): ")
    if response.lower() != 'y':
        print("已取消实验。")
        return
    
    results = []
    
    # 依次运行每个实验
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] ", end="")
        result = run_experiment(exp_config)
        results.append(result)
        
        # 实验间隔
        if i < len(EXPERIMENTS):
            print("\n⏸️  等待 5 秒后开始下一个实验...")
            time.sleep(5)
    
    # 生成报告
    report = generate_report(results)
    
    print("\n\n🎉 所有实验已完成！")
    print(f"📊 成功率: {report['completed']}/{report['total_experiments']} ({report['completed']/report['total_experiments']*100:.1f}%)")

if __name__ == "__main__":
    main()
