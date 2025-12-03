#!/usr/bin/env python3
"""
Plot Results Script

Reads benchmark_summary.json and generates plots.
"""

import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to benchmark_summary.json")
    parser.add_argument("--out", type=str, default="results/plots", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    with open(args.json, "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 1. Bar Chart: Final Average Return per Env/Controller
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="env", y="final_avg_return", hue="controller")
    plt.title("Final Average Return by Environment and Controller")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "return_by_env.png"))
    print(f"Saved {os.path.join(args.out, 'return_by_env.png')}")
    
    # 2. Bar Chart: Wall Time per Env/Controller
    if "wall_time" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="env", y="wall_time", hue="controller")
        plt.title("Training Time (s) by Environment and Controller")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "time_by_env.png"))
        print(f"Saved {os.path.join(args.out, 'time_by_env.png')}")

if __name__ == "__main__":
    main()
