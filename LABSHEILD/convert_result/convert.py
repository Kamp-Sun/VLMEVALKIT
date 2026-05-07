#!/usr/bin/env python3
"""
Convert evaluation results from JSON format to Markdown format.

Usage:
    python convert.py <relative_path_to_summary_json>
    
Example:
    python convert.py ../results/gpt-5.1-2025-11-13/20260124_085207_joint_sall_multi/summary.json
"""

import json
import sys
from pathlib import Path


def convert_json_to_md(json_path: str) -> None:
    """
    Convert a summary.json file to markdown format.
    
    Args:
        json_path: Relative path to the summary.json file
    """
    # Get the absolute path
    json_file = Path(json_path).resolve()
    
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract model name
    model_name = data.get('config', {}).get('model', 'unknown')
    print(f"Processing model: {model_name}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / 'converted_results'
    output_dir.mkdir(exist_ok=True)
    
    # Create markdown file
    md_file = output_dir / f"{model_name}.md"
    
    # Extract metrics
    qa_summary = data.get('qa', {}).get('summary', {})
    prp_metrics = data.get('prp', {}).get('metrics', {})
    
    # Calculate Mean (QA overall accuracy)
    qa_overall = qa_summary.get('overall', {})
    mean_accuracy = qa_overall.get('accuracy', 0.0)
    
    # Define metrics in order with their display names and JSON keys
    metrics_config = [
        # QA Accuracy metrics
        ('Mean', mean_accuracy),
        ('perception.symbol_recognition', qa_summary.get('perception.symbol_recognition', {}).get('accuracy', 0.0)),
        ('perception.spatial_reasoning', qa_summary.get('perception.spatial_reasoning', {}).get('accuracy', 0.0)),
        ('perception.object_recognition', qa_summary.get('perception.object_recognition', {}).get('accuracy', 0.0)),
        ('perception.state_recognition', qa_summary.get('perception.state_recognition', {}).get('accuracy', 0.0)),
        ('reasoning.causal_reasoning', qa_summary.get('reasoning.causal_reasoning', {}).get('accuracy', 0.0)),
        ('reasoning.counterfactual_reasoning', qa_summary.get('reasoning.counterfactual_reasoning', {}).get('accuracy', 0.0)),
        ('planning.next_step_planning', qa_summary.get('planning.next_step_planning', {}).get('accuracy', 0.0)),
        ('planning.action_ordering', qa_summary.get('planning.action_ordering', {}).get('accuracy', 0.0)),
        ('planning.recovery_planning', qa_summary.get('planning.recovery_planning', {}).get('accuracy', 0.0)),
        
        # Label metrics (Jaccard, Precision, Recall)
        ('unsafe_factors_jaccard', prp_metrics.get('unsafe_factors_jaccard', 0.0)),
        ('unsafe_factors_precision', prp_metrics.get('unsafe_factors_precision', 0.0)),
        ('unsafe_factors_recall', prp_metrics.get('unsafe_factors_recall', 0.0)),
        ('hazard_patterns_jaccard', prp_metrics.get('hazard_patterns_jaccard', 0.0)),
        ('hazard_patterns_precision', prp_metrics.get('hazard_patterns_precision', 0.0)),
        ('hazard_patterns_recall', prp_metrics.get('hazard_patterns_recall', 0.0)),
        
        # Planning metrics
        ('plan_judge_analysis_plan_score_avg (ana)', prp_metrics.get('plan_judge_analysis_plan_score_avg', 0.0)),
        ('plan_judge_score_avg', prp_metrics.get('plan_judge_score_avg', 0.0)),
        ('plan_judge_pass_rate', prp_metrics.get('plan_judge_pass_rate', 0.0)),
        
        # Safety level metrics (L23 specific)
        ('safety_level23_accuracy', prp_metrics.get('safety_level23_accuracy', 0.0)),
        ('safety_level23_underestimation_rate', prp_metrics.get('safety_level23_underestimation_rate', 0.0)),
        ('safety_level23_overestimation_rate', prp_metrics.get('safety_level23_overestimation_rate', 0.0)),
    ]
    
    # Generate markdown table content with 2 rows and many columns
    md_lines = []
    
    # Extract metric names and values, format values to 3 decimal places
    metric_names = [name for name, _ in metrics_config]
    metric_values = [f"{float(value):.3f}" for _, value in metrics_config]
    
    # Build header row (metric names)
    header_row = "| " + " | ".join(metric_names) + " |"
    md_lines.append(header_row)
    
    # Build separator row
    separator = "| " + " | ".join(["---"] * len(metric_names)) + " |"
    md_lines.append(separator)
    
    # Build data row (values)
    data_row = "| " + " | ".join(metric_values) + " |"
    md_lines.append(data_row)
    
    # Write to markdown file
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✓ Converted to: {md_file}")


if __name__ == '__main__':
    json_path = sys.argv[1]
    convert_json_to_md(json_path)
