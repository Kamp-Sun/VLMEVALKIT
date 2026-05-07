#!/usr/bin/env python3
"""
Convert all evaluation results from JSON format to a single overall Markdown table.

This script scans the results directory, finds all summary.json files from each model,
and combines them into a single markdown table with all models.

Output: converted_results/overall.md
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def extract_metrics_from_json(data: dict) -> dict:
    """
    Extract all metrics from a summary.json file.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Dictionary with all metric values
    """
    qa_summary = data.get('qa', {}).get('summary', {})
    prp_metrics = data.get('prp', {}).get('metrics', {})
    
    # Calculate Mean (QA overall accuracy)
    qa_overall = qa_summary.get('overall', {})
    mean_accuracy = qa_overall.get('accuracy', 0.0)
    
    # Define metrics in order
    metrics = {
        'Mean': mean_accuracy,
        'perception.symbol_recognition': qa_summary.get('perception.symbol_recognition', {}).get('accuracy', 0.0),
        'perception.spatial_reasoning': qa_summary.get('perception.spatial_reasoning', {}).get('accuracy', 0.0),
        'perception.object_recognition': qa_summary.get('perception.object_recognition', {}).get('accuracy', 0.0),
        'perception.state_recognition': qa_summary.get('perception.state_recognition', {}).get('accuracy', 0.0),
        'reasoning.causal_reasoning': qa_summary.get('reasoning.causal_reasoning', {}).get('accuracy', 0.0),
        'reasoning.counterfactual_reasoning': qa_summary.get('reasoning.counterfactual_reasoning', {}).get('accuracy', 0.0),
        'planning.next_step_planning': qa_summary.get('planning.next_step_planning', {}).get('accuracy', 0.0),
        'planning.action_ordering': qa_summary.get('planning.action_ordering', {}).get('accuracy', 0.0),
        'planning.recovery_planning': qa_summary.get('planning.recovery_planning', {}).get('accuracy', 0.0),
        'unsafe_factors_jaccard': prp_metrics.get('unsafe_factors_jaccard', 0.0),
        'unsafe_factors_precision': prp_metrics.get('unsafe_factors_precision', 0.0),
        'unsafe_factors_recall': prp_metrics.get('unsafe_factors_recall', 0.0),
        'hazard_patterns_jaccard': prp_metrics.get('hazard_patterns_jaccard', 0.0),
        'hazard_patterns_precision': prp_metrics.get('hazard_patterns_precision', 0.0),
        'hazard_patterns_recall': prp_metrics.get('hazard_patterns_recall', 0.0),
        'plan_judge_analysis_plan_score_avg (ana)': prp_metrics.get('plan_judge_analysis_plan_score_avg', 0.0),
        'plan_judge_score_avg': prp_metrics.get('plan_judge_score_avg', 0.0),
        'plan_judge_pass_rate': prp_metrics.get('plan_judge_pass_rate', 0.0),
        'safety_level23_accuracy': prp_metrics.get('safety_level23_accuracy', 0.0),
        'safety_level23_underestimation_rate': prp_metrics.get('safety_level23_underestimation_rate', 0.0),
        'safety_level23_overestimation_rate': prp_metrics.get('safety_level23_overestimation_rate', 0.0),
    }
    
    return metrics


def find_latest_summary_json(results_dir: Path, model_name: str) -> Path:
    """
    Find the latest summary.json for a given model.
    Only considers subdirectories that contain "all" in their name.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model
        
    Returns:
        Path to the summary.json file or None if not found
    """
    model_dir = results_dir / model_name
    
    if not model_dir.exists():
        return None
    
    # Find all timestamped subdirectories that contain "all" in their name
    subdirs = [d for d in model_dir.iterdir() if d.is_dir() and 'all' in d.name.lower()]
    
    if not subdirs:
        return None
    
    # Sort by modification time and get the latest
    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime)[-1]
    
    summary_file = latest_dir / 'summary.json'
    if summary_file.exists():
        return summary_file
    
    return None


def convert_all_to_overall_md(results_path: str = '../results') -> None:
    """
    Convert all model results to a single overall markdown table.
    
    Args:
        results_path: Relative path to the results directory
    """
    results_dir = Path(results_path).resolve()
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Get all model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    model_names = sorted([d.name for d in model_dirs])
    
    print(f"Found {len(model_names)} models in {results_dir}")
    
    # Collect data for all models
    all_data = []
    metric_names = None
    
    for model_name in model_names:
        print(f"  Processing: {model_name}")
        
        summary_file = find_latest_summary_json(results_dir, model_name)
        
        if not summary_file:
            print(f"    ⚠ No summary.json found for {model_name}, skipping")
            continue
        
        # Load and extract metrics
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metrics = extract_metrics_from_json(data)
            
            # Skip models with Mean accuracy of 0
            mean_accuracy = metrics.get('Mean', 0.0)
            if mean_accuracy == 0.0:
                print(f"    ⚠ Skipping {model_name}: Mean accuracy is 0")
                continue
            
            if metric_names is None:
                metric_names = list(metrics.keys())
            
            all_data.append({
                'model': model_name,
                'metrics': metrics
            })
            
            print(f"    ✓ Processed successfully")
        
        except Exception as e:
            print(f"    ✗ Error processing {summary_file}: {e}")
            continue
    
    if not all_data:
        print("Error: No data to process")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / 'converted_results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate markdown table
    md_lines = []
    
    # Build header row with model column first
    header_row = "| model | " + " | ".join(metric_names) + " |"
    md_lines.append(header_row)
    
    # Build separator row
    separator = "| --- | " + " | ".join(["---"] * len(metric_names)) + " |"
    md_lines.append(separator)
    
    # Build data rows
    for row_data in all_data:
        model_name = row_data['model']
        metrics = row_data['metrics']
        
        # Format metric values to 3 decimal places
        metric_values = [f"{float(metrics[m]):.3f}" for m in metric_names]
        
        data_row = "| " + model_name + " | " + " | ".join(metric_values) + " |"
        md_lines.append(data_row)
    
    # Write to markdown file
    output_file = output_dir / 'overall.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✓ Generated overall results table: {output_file}")
    print(f"  Total models: {len(all_data)}")
    print(f"  Total metrics: {len(metric_names)}")


if __name__ == '__main__':
    results_path = sys.argv[1] if len(sys.argv) > 1 else '../final_result'
    convert_all_to_overall_md(results_path)
