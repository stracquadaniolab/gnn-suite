#!/usr/bin/env python3


import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

"""
compare_line_plots.py - Compare GNN Training Metrics Line Plots

This script reads GNN training metrics from data files and generates line plots
to compare the training progress of different models over epochs.

Usage:
    python compare_line_plots.py --line_plot <output_file> --folder <data_folder> --base_name <base_name>
    python bin/compare_models_local.py --line_plot results/comparison/string_model_comp.pdf --folder results/data/string --base_name test.txt

Arguments:
    --line_plot <output_file>
        Path to save the generated line plot (PDF format recommended).

    --folder <data_folder>
        Location of the data files containing GNN training metrics.
    
    --base_name <base_name>
        Base name of the files to be analyzed. The script will search for files
        that start with this base name and have a specific naming convention.

Example:
    python compare_line_plots.py --line_plot line_plot.pdf --folder ../results/data --base_name full-gat3h-300

Note:
    This script extracts model names, epochs, and runs from the file names
    in the specified folder based on a specific naming convention. It then
    generates line plots to compare the training progress of the models.
"""


def plot_line_plots(line_plot, data_files, metrics, model_names, runs):
    # Read data from the files
    all_data = []
    for file, model_name, run in zip(data_files, model_names, runs):
        df = pd.read_table(file, delimiter="\s+", comment='#')
        df['model'] = model_name
        df['run'] = run
        all_data.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(all_data, ignore_index=True)
    
    # print(all_data)

    sns.set(style='whitegrid', font_scale=1.6)
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    
    num_plots = len(metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16), sharex=True)
    axes = axes.flatten()
    
    # Define a color palette with distinct colors for each model
    palette = sns.color_palette("tab10", len(set(model_names)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Using sns.lineplot to plot the mean and standard deviation of metrics
        sns.lineplot(data=all_data, x='epoch', y=metric, hue='model', errorbar='se', ax=ax, palette=palette)
            
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=16, fontweight='bold')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [label.upper() for label in labels])
        
    fig.suptitle("GNN Training Progress", fontsize=22, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(line_plot, dpi=300)
    plt.close()
    print(f"Line plots saved as {line_plot}")


def extract_info_from_file_name(file_name):
    parts = os.path.splitext(file_name)[0].split('-')
    print("Parts:")
    print(parts)
    parts = parts[2:]
    print("Sliced parts:")
    print(parts)
    model_name, epoch_str, run_prefix, run_str, base_name = parts[:5]
    epoch = int(epoch_str)
    run = int(run_str)  
    
    return base_name, model_name, epoch, run

def max_bacc_stats(filename, data_files, model_names, runs):
    # Read data from the files
    all_data = []
    for file, model_name, run in zip(data_files, model_names, runs):
        df = pd.read_table(file, delimiter="\s+", comment='#')
        df['model'] = model_name
        df['run'] = run
        all_data.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(all_data, ignore_index=True)
    
    # Group by model and epoch, then compute mean and std of bacc for each group
    grouped = all_data.groupby(['model', 'epoch'])['bacc'].agg(['mean', 'std']).reset_index()
    
    # Identify the epoch with max mean bacc for each model
    idx = grouped.groupby('model')['mean'].idxmax()
    max_bacc_stats = grouped.loc[idx]
    
    # Combine mean and std in one column with "±" symbol
   # max_bacc_stats['mean ± sems'] = max_bacc_stats['mean'].round(3).astype(str) + " ± " + max_bacc_stats['sems'].round(3).astype(str)
    max_bacc_stats['mean ± std'] = max_bacc_stats['mean'].round(3).astype(str) + " ± " + max_bacc_stats['std'].round(3).astype(str)

    # Save the stats to a file
    max_bacc_stats.to_latex(filename, index=False, columns=['model','epoch', 'mean ± std'])
    print(f"Stats saved as {filename}")
    print(max_bacc_stats[['model', 'epoch', 'mean ± std']])
    return max_bacc_stats



def main():
    parser = argparse.ArgumentParser(description='Compare GNN training metrics')
    parser.add_argument('--line_plot', type=str, help='Path to save the line plot')
    parser.add_argument('--folder', type=str, default='../results/data', help='Location of the data files (default: ./results/)')
    parser.add_argument('--base_name', type=str, help='Base name for extracting information')

    args = parser.parse_args()
    args.line_plot = os.path.abspath(args.line_plot)
    args.folder = os.path.abspath(args.folder)
   
    data_files = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if file.endswith(args.base_name)]
    
    file_info = [extract_info_from_file_name(file) for file in data_files]
    base_names, model_names, epochs, runs = zip(*file_info)
    print(runs)
   

    metrics = ['loss', 'prec', 'rec', 'acc', 'bacc', 'auc']

    plot_line_plots(args.line_plot, data_files, metrics, model_names, runs)

    # model_names = ["gcn2", "gcn", "gat", "gat3h", "hgcn", "phgcn", "sage", "gin", "gtn"]
    #max_bacc = max_bacc_stats(args.line_plot, data_files, model_names, runs)

if __name__ == "__main__":
    main()



