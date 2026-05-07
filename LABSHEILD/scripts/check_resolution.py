#!/usr/bin/env python3
"""
检查所有模型评测时使用的分辨率配置
"""

import sys
from pathlib import Path

# 添加 vlmeval 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlmeval.config import supported_VLM
import inspect


def get_resolution_info(model_name: str, model_func):
    """获取模型的分辨率信息"""
    info = {
        "model_name": model_name,
        "resolution": "未知",
        "min_pixels": None,
        "max_pixels": None,
        "img_size": None,
        "img_detail": None,
    }
    
    try:
        # 获取 partial 函数的参数
        if hasattr(model_func, 'func') and hasattr(model_func, 'keywords'):
            # 这是一个 functools.partial 对象
            keywords = model_func.keywords
            info["min_pixels"] = keywords.get("min_pixels")
            info["max_pixels"] = keywords.get("max_pixels")
            info["img_size"] = keywords.get("img_size")
            info["img_detail"] = keywords.get("img_detail")
            
            # 计算分辨率
            if info["min_pixels"] and info["max_pixels"]:
                # Qwen2-VL 格式: min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28
                # 这表示最小 1280x1280，最大 16384x16384（以 patch 28x28 为单位）
                if isinstance(info["min_pixels"], int) and isinstance(info["max_pixels"], int):
                    min_res = int((info["min_pixels"] / (28 * 28)) ** 0.5)
                    max_res = int((info["max_pixels"] / (28 * 28)) ** 0.5)
                    info["resolution"] = f"{min_res}x{min_res} ~ {max_res}x{max_res}"
                else:
                    info["resolution"] = f"min_pixels={info['min_pixels']}, max_pixels={info['max_pixels']}"
            elif info["img_size"]:
                if info["img_size"] == -1:
                    info["resolution"] = "高分辨率（无限制）"
                else:
                    info["resolution"] = f"{info['img_size']}x{info['img_size']}"
                if info["img_detail"]:
                    info["resolution"] += f" ({info['img_detail']})"
            elif info["min_pixels"] or info["max_pixels"]:
                info["resolution"] = f"min_pixels={info['min_pixels']}, max_pixels={info['max_pixels']}"
            else:
                # 检查模型类的默认值
                func = model_func.func if hasattr(model_func, 'func') else model_func
                if hasattr(func, '__name__'):
                    class_name = func.__name__
                    # Qwen3-VL 默认不设置分辨率参数，使用模型默认值
                    if "Qwen3" in class_name:
                        info["resolution"] = "模型默认（通常支持动态分辨率）"
                    elif "GPT4V" in class_name or "GPT4o" in class_name:
                        info["resolution"] = "API 默认（通常 512x512 或高分辨率）"
                    else:
                        info["resolution"] = "模型默认"
    except Exception as e:
        info["error"] = str(e)
    
    return info


def main():
    print("=" * 80)
    print("模型分辨率配置检查")
    print("=" * 80)
    print()
    
    # 按系列分组
    model_series = {}
    for model_name in sorted(supported_VLM.keys()):
        model_func = supported_VLM[model_name]
        info = get_resolution_info(model_name, model_func)
        
        # 根据模型名称判断系列
        series = "其他"
        if "Qwen3-VL" in model_name or "Qwen3-Omni" in model_name:
            series = "Qwen3-VL"
        elif "Qwen2" in model_name or "Qwen-VL" in model_name:
            series = "Qwen2-VL"
        elif "GPT4" in model_name or "GPT4o" in model_name or "gpt-5" in model_name:
            series = "GPT 系列"
        elif "Claude" in model_name:
            series = "Claude 系列"
        elif "Gemini" in model_name:
            series = "Gemini 系列"
        elif "HunYuan" in model_name:
            series = "HunYuan 系列"
        elif "InternVL" in model_name:
            series = "InternVL 系列"
        elif "RoboBrain" in model_name:
            series = "RoboBrain 系列"
        
        if series not in model_series:
            model_series[series] = []
        model_series[series].append(info)
    
    # 打印结果
    for series in sorted(model_series.keys()):
        models = model_series[series]
        print(f"\n## {series} ({len(models)} 个模型)")
        print("-" * 80)
        print(f"{'模型名称':<40} {'分辨率配置':<40}")
        print("-" * 80)
        
        for info in models:
            model_name = info["model_name"]
            resolution = info["resolution"]
            if len(model_name) > 38:
                model_name = model_name[:35] + "..."
            if len(resolution) > 38:
                resolution = resolution[:35] + "..."
            print(f"{model_name:<40} {resolution:<40}")
    
    # 统计
    print("\n" + "=" * 80)
    print("分辨率配置统计")
    print("=" * 80)
    
    resolution_types = {}
    for series in model_series.values():
        for info in series:
            res = info["resolution"]
            resolution_types[res] = resolution_types.get(res, 0) + 1
    
    for res_type, count in sorted(resolution_types.items(), key=lambda x: -x[1]):
        print(f"{res_type}: {count} 个模型")
    
    # 详细列出有明确分辨率设置的模型
    print("\n" + "=" * 80)
    print("有明确分辨率设置的模型")
    print("=" * 80)
    
    for series in sorted(model_series.keys()):
        models = model_series[series]
        has_explicit_res = [m for m in models if m["min_pixels"] or m["max_pixels"] or m["img_size"]]
        if has_explicit_res:
            print(f"\n{series}:")
            for info in has_explicit_res:
                details = []
                if info["min_pixels"]:
                    details.append(f"min_pixels={info['min_pixels']}")
                if info["max_pixels"]:
                    details.append(f"max_pixels={info['max_pixels']}")
                if info["img_size"]:
                    details.append(f"img_size={info['img_size']}")
                if info["img_detail"]:
                    details.append(f"img_detail={info['img_detail']}")
                print(f"  - {info['model_name']}: {', '.join(details)}")


if __name__ == "__main__":
    main()

