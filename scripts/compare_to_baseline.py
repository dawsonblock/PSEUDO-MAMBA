#!/usr/bin/env python3
"""
Compare user benchmark results to reference baseline.

Usage:
    python scripts/compare_to_baseline.py --results results/benchmark_summary.json
"""

import argparse
import json
from typing import Dict, List, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON results file."""
    with open(path, 'r') as f:
        data = json.load(f)
        # Handle both list format and dict with "results" key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "results" in data:
            return data["results"]
        else:
            raise ValueError(f"Unexpected JSON format in {path}")


def find_matching_baseline(user_result: Dict[str, Any], baseline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the matching baseline result for a user result."""
    for baseline in baseline_results:
        if (baseline["task"] == user_result["task"] and
            baseline["controller"] == user_result["controller"] and
            baseline.get("horizon") == user_result.get("horizon")):
            return baseline
    return None


def compute_diff(user_val: float, baseline_val: float) -> Dict[str, Any]:
    """Compute difference metrics."""
    abs_diff = user_val - baseline_val
    if baseline_val != 0:
        pct_diff = (abs_diff / baseline_val) * 100
    else:
        pct_diff = 0.0
    
    return {
        "abs_diff": abs_diff,
        "pct_diff": pct_diff
    }


def print_comparison_table(user_results: List[Dict[str, Any]], baseline_results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON TO REFERENCE BASELINE")
    print("=" * 100)
    
    print(f"\n{'Task':<15} {'Ctrl':<12} {'H':<6} {'User Return':>12} {'Baseline':>12} {'Diff':>10} {'Status':<10}")
    print("-" * 100)
    
    total_comparisons = 0
    better_count = 0
    worse_count = 0
    
    for user_result in user_results:
        baseline = find_matching_baseline(user_result, baseline_results)
        
        if baseline is None:
            status = "NO BASELINE"
            diff_str = "N/A"
        else:
            total_comparisons += 1
            user_return = user_result.get("final_avg_return", 0.0)
            baseline_return = baseline.get("final_avg_return", 0.0)
            
            diff_data = compute_diff(user_return, baseline_return)
            abs_diff = diff_data["abs_diff"]
            pct_diff = diff_data["pct_diff"]
            
            diff_str = f"{abs_diff:+.3f} ({pct_diff:+.1f}%)"
            
            # Determine status
            if abs_diff > 0.05:  # Significantly better
                status = "✓ BETTER"
                better_count += 1
            elif abs_diff < -0.05:  # Significantly worse
                status = "✗ WORSE"
                worse_count += 1
            else:
                status = "≈ SIMILAR"
        
        user_return_str = f"{user_result.get('final_avg_return', 0.0):.3f}"
        baseline_return_str = f"{baseline.get('final_avg_return', 0.0):.3f}" if baseline else "N/A"
        
        print(f"{user_result['task']:<15} {user_result['controller']:<12} "
              f"{user_result.get('horizon', 'N/A'):<6} {user_return_str:>12} "
              f"{baseline_return_str:>12} {diff_str:>10} {status:<10}")
    
    print("-" * 100)
    print(f"\nSummary: {total_comparisons} comparisons | "
          f"{better_count} better | {worse_count} worse | "
          f"{total_comparisons - better_count - worse_count} similar")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results to baseline")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to user benchmark results JSON")
    parser.add_argument("--baseline", type=str,
                        default="results/reference_metrics.json",
                        help="Path to reference baseline JSON")
    args = parser.parse_args()
    
    try:
        user_results = load_json(args.results)
        baseline_results = load_json(args.baseline)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure both files exist:")
        print(f"  User results: {args.results}")
        print(f"  Baseline: {args.baseline}")
        return 1
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return 1
    
    print_comparison_table(user_results, baseline_results)
    return 0


if __name__ == "__main__":
    exit(main())
