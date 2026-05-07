#!/usr/bin/env python3
"""
生成按等级分组的指标报告（Markdown 格式）
"""

import json
from pathlib import Path
from typing import Dict, Any


def generate_markdown_report(metrics_file: str, output_file: str):
    """生成 Markdown 格式的报告"""
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    report_lines = []
    report_lines.append("# 按操作等级和安全等级的指标报告\n")
    report_lines.append("本报告展示了不同模型在不同操作等级（operation_level）和安全等级（safety_level）下的性能指标。\n")
    
    # 按操作等级汇总
    report_lines.append("## 按操作等级 (Operation Level) 汇总\n")
    report_lines.append("| 模型 | Operation Level | QA准确率 | PRP安全等级准确率 | PRP不安全因素Jaccard | PRP危险模式Jaccard |\n")
    report_lines.append("|------|----------------|----------|-------------------|---------------------|-------------------|\n")
    
    for model_name in sorted(all_results.keys()):
        result = all_results[model_name]
        by_op = result.get("by_operation_level", {})
        for op_level in sorted(by_op.keys()):
            data = by_op[op_level]
            qa_acc = data["qa"]["accuracy"]
            prp_safety_acc = data["prp"]["safety_level_accuracy"]
            prp_unsafe_j = data["prp"]["unsafe_factors_jaccard"]
            prp_hazard_j = data["prp"]["hazard_patterns_jaccard"]
            report_lines.append(
                f"| {model_name} | {op_level} | {qa_acc:.3f} ({data['qa']['correct']}/{data['qa']['total']}) | "
                f"{prp_safety_acc:.3f} ({data['prp']['safety_level_correct']}/{data['prp']['safety_level_total']}) | "
                f"{prp_unsafe_j:.3f} | {prp_hazard_j:.3f} |\n"
            )
    
    # 按安全等级汇总
    report_lines.append("\n## 按安全等级 (Safety Level) 汇总\n")
    report_lines.append("| 模型 | Safety Level | QA准确率 | PRP安全等级准确率 | PRP不安全因素Jaccard | PRP危险模式Jaccard |\n")
    report_lines.append("|------|--------------|----------|-------------------|---------------------|-------------------|\n")
    
    for model_name in sorted(all_results.keys()):
        result = all_results[model_name]
        by_safety = result.get("by_safety_level", {})
        for safety_level in sorted(by_safety.keys()):
            data = by_safety[safety_level]
            qa_acc = data["qa"]["accuracy"]
            prp_safety_acc = data["prp"]["safety_level_accuracy"]
            prp_unsafe_j = data["prp"]["unsafe_factors_jaccard"]
            prp_hazard_j = data["prp"]["hazard_patterns_jaccard"]
            report_lines.append(
                f"| {model_name} | {safety_level} | {qa_acc:.3f} ({data['qa']['correct']}/{data['qa']['total']}) | "
                f"{prp_safety_acc:.3f} ({data['prp']['safety_level_correct']}/{data['prp']['safety_level_total']}) | "
                f"{prp_unsafe_j:.3f} | {prp_hazard_j:.3f} |\n"
            )
    
    # 详细统计（按模型分组）
    report_lines.append("\n## 详细统计（按模型分组）\n")
    
    for model_name in sorted(all_results.keys()):
        result = all_results[model_name]
        report_lines.append(f"\n### {model_name}\n")
        
        # 操作等级
        report_lines.append("#### 按操作等级\n")
        report_lines.append("| Operation Level | QA准确率 | PRP安全等级准确率 | PRP不安全因素Jaccard | PRP危险模式Jaccard |\n")
        report_lines.append("|----------------|----------|-------------------|---------------------|-------------------|\n")
        by_op = result.get("by_operation_level", {})
        for op_level in sorted(by_op.keys()):
            data = by_op[op_level]
            qa_acc = data["qa"]["accuracy"]
            prp_safety_acc = data["prp"]["safety_level_accuracy"]
            prp_unsafe_j = data["prp"]["unsafe_factors_jaccard"]
            prp_hazard_j = data["prp"]["hazard_patterns_jaccard"]
            report_lines.append(
                f"| {op_level} | {qa_acc:.3f} ({data['qa']['correct']}/{data['qa']['total']}) | "
                f"{prp_safety_acc:.3f} ({data['prp']['safety_level_correct']}/{data['prp']['safety_level_total']}) | "
                f"{prp_unsafe_j:.3f} | {prp_hazard_j:.3f} |\n"
            )
        
        # 安全等级
        report_lines.append("\n#### 按安全等级\n")
        report_lines.append("| Safety Level | QA准确率 | PRP安全等级准确率 | PRP不安全因素Jaccard | PRP危险模式Jaccard |\n")
        report_lines.append("|--------------|----------|-------------------|---------------------|-------------------|\n")
        by_safety = result.get("by_safety_level", {})
        for safety_level in sorted(by_safety.keys()):
            data = by_safety[safety_level]
            qa_acc = data["qa"]["accuracy"]
            prp_safety_acc = data["prp"]["safety_level_accuracy"]
            prp_unsafe_j = data["prp"]["unsafe_factors_jaccard"]
            prp_hazard_j = data["prp"]["hazard_patterns_jaccard"]
            report_lines.append(
                f"| {safety_level} | {qa_acc:.3f} ({data['qa']['correct']}/{data['qa']['total']}) | "
                f"{prp_safety_acc:.3f} ({data['prp']['safety_level_correct']}/{data['prp']['safety_level_total']}) | "
                f"{prp_unsafe_j:.3f} | {prp_hazard_j:.3f} |\n"
            )
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"报告已生成: {output_file}")


def main():
    script_dir = Path(__file__).parent
    metrics_file = script_dir.parent / "data" / "metrics_by_levels.json"
    output_file = script_dir.parent / "data" / "metrics_by_levels_report.md"
    
    if not metrics_file.exists():
        print(f"错误: 找不到指标文件 {metrics_file}")
        print("请先运行 scripts/analyze_by_levels.py 生成指标文件")
        return
    
    generate_markdown_report(str(metrics_file), str(output_file))


if __name__ == "__main__":
    main()

