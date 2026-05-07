#!/usr/bin/env python3
"""
提取没有透明物体标签的样本，生成测试数据集
"""

import json
from pathlib import Path


def extract_samples_without_fragile(
    dataset_path: str = "data/dataset.json",
    output_path: str = "data/dataset_without_fragile.json"
):
    """提取没有透明物体标签的样本"""
    
    dataset_file = Path(__file__).parent.parent / dataset_path
    output_file = Path(__file__).parent.parent / output_path
    
    # 读取数据集
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 读取之前统计的结果，获取样本 ID 列表
    stats_file = Path(__file__).parent.parent / "data" / "samples_without_fragile.json"
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        sample_ids_without_fragile = set(stats["samples_without_fragile_list"])
    else:
        # 如果没有统计文件，重新计算
        sample_ids_without_fragile = set()
        for sample in dataset:
            sample_id = sample.get("sample_id", "")
            prp = sample.get("open_prp") or sample.get("prp", {})
            unsafe_factors = prp.get("perception", {}).get("unsafe_factors", [])
            if "Fragile_Material_Presence" not in unsafe_factors:
                sample_ids_without_fragile.add(sample_id)
    
    # 提取样本
    extracted_samples = []
    for sample in dataset:
        sample_id = sample.get("sample_id", "")
        if sample_id in sample_ids_without_fragile:
            extracted_samples.append(sample)
    
    # 保存到新文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_samples, f, indent=2, ensure_ascii=False)
    
    print(f"已提取 {len(extracted_samples)} 个没有透明物体标签的样本")
    print(f"保存到: {output_file}")
    
    # 打印一些统计信息
    from collections import Counter
    scenario_cnt = Counter()
    op_level_cnt = Counter()
    safety_level_cnt = Counter()
    
    for sample in extracted_samples:
        scenario_cnt[sample.get("scenario", "Unknown")] += 1
        op_level_cnt[sample.get("operation_level", -1)] += 1
        safety_level_cnt[sample.get("safety_level", -1)] += 1
    
    print("\n提取的样本统计:")
    print(f"  总样本数: {len(extracted_samples)}")
    print(f"\n按场景分布:")
    for scenario in sorted(scenario_cnt.keys()):
        print(f"  {scenario}: {scenario_cnt[scenario]} 个")
    print(f"\n按操作等级分布:")
    for op_level in sorted(op_level_cnt.keys()):
        print(f"  Operation Level {op_level}: {op_level_cnt[op_level]} 个")
    print(f"\n按安全等级分布:")
    for safety_level in sorted(safety_level_cnt.keys()):
        print(f"  Safety Level {safety_level}: {safety_level_cnt[safety_level]} 个")
    
    return extracted_samples


if __name__ == "__main__":
    extract_samples_without_fragile()

