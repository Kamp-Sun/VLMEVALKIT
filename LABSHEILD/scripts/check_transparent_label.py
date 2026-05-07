#!/usr/bin/env python3
"""
统计数据集中没有透明物体标签（Fragile_Material_Presence）的样本数量
"""

import json
from pathlib import Path
from collections import Counter


def check_transparent_labels(dataset_path: str = "data/dataset.json"):
    """检查数据集中透明物体标签的分布"""
    
    dataset_file = Path(__file__).parent.parent / dataset_path
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_samples = len(dataset)
    samples_with_fragile = 0
    samples_without_fragile = 0
    samples_with_fragile_list = []
    samples_without_fragile_list = []
    
    # 统计按场景和操作等级分布
    fragile_by_scenario = Counter()
    fragile_by_op_level = Counter()
    no_fragile_by_scenario = Counter()
    no_fragile_by_op_level = Counter()
    
    for sample in dataset:
        sample_id = sample.get("sample_id", "")
        scenario = sample.get("scenario", "Unknown")
        op_level = sample.get("operation_level", -1)
        
        # 检查 unsafe_factors 中是否有 Fragile_Material_Presence
        prp = sample.get("open_prp") or sample.get("prp", {})
        unsafe_factors = prp.get("perception", {}).get("unsafe_factors", [])
        
        has_fragile = "Fragile_Material_Presence" in unsafe_factors
        
        if has_fragile:
            samples_with_fragile += 1
            samples_with_fragile_list.append(sample_id)
            fragile_by_scenario[scenario] += 1
            fragile_by_op_level[op_level] += 1
        else:
            samples_without_fragile += 1
            samples_without_fragile_list.append(sample_id)
            no_fragile_by_scenario[scenario] += 1
            no_fragile_by_op_level[op_level] += 1
    
    # 打印结果
    print("=" * 80)
    print("透明物体标签（Fragile_Material_Presence）统计")
    print("=" * 80)
    print()
    print(f"总样本数: {total_samples}")
    print(f"有透明物体标签的样本: {samples_with_fragile} ({samples_with_fragile/total_samples*100:.1f}%)")
    print(f"没有透明物体标签的样本: {samples_without_fragile} ({samples_without_fragile/total_samples*100:.1f}%)")
    print()
    
    print("=" * 80)
    print("按场景分布（有透明物体标签）")
    print("=" * 80)
    for scenario in sorted(fragile_by_scenario.keys()):
        count = fragile_by_scenario[scenario]
        print(f"  {scenario}: {count} 个样本")
    
    print()
    print("=" * 80)
    print("按场景分布（没有透明物体标签）")
    print("=" * 80)
    for scenario in sorted(no_fragile_by_scenario.keys()):
        count = no_fragile_by_scenario[scenario]
        print(f"  {scenario}: {count} 个样本")
    
    print()
    print("=" * 80)
    print("按操作等级分布（有透明物体标签）")
    print("=" * 80)
    for op_level in sorted(fragile_by_op_level.keys()):
        count = fragile_by_op_level[op_level]
        print(f"  Operation Level {op_level}: {count} 个样本")
    
    print()
    print("=" * 80)
    print("按操作等级分布（没有透明物体标签）")
    print("=" * 80)
    for op_level in sorted(no_fragile_by_op_level.keys()):
        count = no_fragile_by_op_level[op_level]
        print(f"  Operation Level {op_level}: {count} 个样本")
    
    # 保存详细列表
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "samples_without_fragile.json", 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": total_samples,
            "samples_with_fragile": samples_with_fragile,
            "samples_without_fragile": samples_without_fragile,
            "samples_without_fragile_list": samples_without_fragile_list,
            "samples_with_fragile_list": samples_with_fragile_list,
            "by_scenario": {
                "with_fragile": dict(fragile_by_scenario),
                "without_fragile": dict(no_fragile_by_scenario),
            },
            "by_op_level": {
                "with_fragile": dict(fragile_by_op_level),
                "without_fragile": dict(no_fragile_by_op_level),
            }
        }, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"详细结果已保存到: {output_dir / 'samples_without_fragile.json'}")
    
    return {
        "total": total_samples,
        "with_fragile": samples_with_fragile,
        "without_fragile": samples_without_fragile,
    }


if __name__ == "__main__":
    check_transparent_labels()

