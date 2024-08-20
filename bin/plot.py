#!/usr/bin/env python3
import typer
from pathlib import Path
from typing import List 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_epochs(filename: str, data: List[Path], metric='loss'):
    dfs = [pd.read_table(f, delimiter="\s+", comment='#') for f in data]
    data = pd.concat(dfs)
    
    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=data, x='epoch', y=metric, markers=True, ax=ax)
    ax.set(xlabel='Epoch', ylabel=metric.capitalize())
    ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=16, fontweight='bold')
    ax.set_xlim(1, max(data['epoch']))
    
    fig.suptitle("GNN Training Progress", fontsize=20, fontweight='bold', y=0.95)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")




def plot_metrics(filename: str, data: List[Path], 
                 metrics=('loss', 'prec', 'rec', 'acc', 'bacc', 'auc'), model: str = ''):
    dfs = [pd.read_table(f, delimiter="\s+", comment='#') for f in data]
    data = pd.concat(dfs)

    print(data.head())

    sns.set(style='whitegrid', font_scale=1.2, font="Times New Roman")
    mpl.rcParams['mathtext.fontset'] = 'cm'  # Use LaTeX font for math symbols
    
    num_plots = len(metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    
    # Define a color palette with distinct colors for each subplot
    palette = sns.color_palette("tab10", num_plots)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        color = palette[i % len(palette)]  # Get a distinct color from the palette

        # Calculate mean and standard deviation for the metric
  
        
        # Plot the mean line
        sns.lineplot(data=data, x='epoch', y=metric, markers=False,
                    ax=ax, color=color, label=metric.capitalize())
        
        # Plot the error bars
       # ax.fill_between(metric_mean.index, metric_mean - metric_std,
       #                 metric_mean + metric_std, color=color, alpha=0.2)
        
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=14, fontweight='bold')
        ax.set_xlim(1, max(data['epoch']))
        
    fig.suptitle("GNN Training Progress", fontsize=16, fontweight='bold', y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Add model information to the plot
    plt.figtext(0.5, 0.02, f"Model: {model}", ha='center', fontsize=12)
    
    plt.savefig(filename)
    plt.close()
    print(f"Plots saved as {filename}")


def main(
    filename: str,  # Filename for single plot or metrics plot
    data: List[Path],  # List of data file paths
    metric: str = 'loss',  # Metric to plot (default: 'loss')
    metrics: List[str] = ['loss', 'prec', 'rec', 'acc', 'bacc', 'auc'],  # Metrics to plot for metrics plot
    model: str = '',  # Model information
):
    if len(data) == 1:
        plot_epochs(filename, data, metric)
    else:
        # print all variables passed in
        print(f"filename: {filename}")
        print(f"data: {data}")
        print(f"metric: {metric}")
        print(f"metrics: {metrics}")
        print(f"model: {model}")

        plot_metrics(filename, data, metrics, model)


if __name__ == "__main__":
    typer.run(main)





