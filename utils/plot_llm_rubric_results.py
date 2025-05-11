import sys, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# metrics = ["Clarity", "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness"]

def plot_llm_rubric(fpath):
    """
    Data example:
                    Clarity  Engagement  Coherence  Depth  Relevance  Progress  Naturalness  Truthfulness
    LLMs                                                                                               
    deepseek-llm     4.40        4.06       4.09   3.83       4.30      4.35         4.30          3.98
    qwen2            4.89        3.97       5.00   3.37       4.90      4.99         4.00          3.71
    gemma            3.96        3.14       3.98   3.19       4.09      3.97         3.26          3.91
    olmo2            4.88        4.43       4.87   4.46       4.88      4.88         4.62          4.87
    openchat         4.99        4.61       4.99   4.34       5.00      4.98         4.84          4.65
    llama3           4.67        3.89       4.69   3.44       4.73      4.68         3.87          3.78
    phi4             4.66        3.94       4.60   3.57       4.92      4.52         3.96          2.73
    """
    df = pd.read_csv(fpath, index_col='LLMs')
    print(df)

    markers = ['o', 's', 'X', 'D', 'P', '*', '^', 'v', '<', '>']  # Circle, square, diamond, triangle, etc.
    colors = plt.cm.get_cmap('tab10', len(df.columns))     # Use a colormap for different colors

    num_categories = len(df.columns)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # Make the plot circular by repeating the first angle at the end
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Prepare the x-ticks: [Topic1, Topic2, ...] with angle wrapping
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{df.columns[i]}" for i in range(num_categories)], fontsize=8, ha='center', va='center') # 'top', 'bottom', 'center', 'baseline', 'center_baseline
    ax.tick_params(axis='x', pad=11)  # Increase vertical distance from axis

    # Plot each model as a line
    for i, model in enumerate(df.index):
        values = df.iloc[i].tolist()
        values += values[:1]  # close the loop
        ax.plot(angles, values, label=model, marker=markers[i % len(markers)],
        color=colors(i),
        linewidth=2,
        markersize=6)

    # Add labels to radial axis
    ax.set_ylim(1, 5.35)
    scale = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ax.set_yticks(scale)
    ax.set_yticklabels(scale)

    # Optional: annotate topic names around the circle
    # for i, label in enumerate(df.columns):
    #     angle = angles[i]
    #     ax.text(angle, 5.4, f"{label}", ha='center', va='center', fontsize=6)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.08), fontsize=8)

    # Save the plot
    plt.tight_layout()
    plt.savefig(fpath.replace('result', 'figure').replace('.csv','.png'), dpi=400)
    plt.close()

if __name__ == "__main__":
    # file_tasks_per_topic = 'result/LLM_rubric_result_homo.csv'
    if len(sys.argv)==2:
        """
        Usage example:
        python utils/plot_llm_rubric_results.py result/LLM_rubric_result_homo.csv
        """
        plot_llm_rubric(fpath=sys.argv[1])
    elif len(sys.argv)==1:
        fpaths = [
            'LLM_rubric_result_homo.csv',
            'LLM_rubric_result_heter.csv'
        ]
        for fpath in fpaths:
            plot_llm_rubric(fpath=f"result/{fpath}")