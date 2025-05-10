import sys, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

metrics = ["Clarity", "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness"]

def plot_tasks_per_topic(df, fpath):
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P']  # Circle, square, diamond, triangle, etc.
    colors = plt.cm.get_cmap('tab10', len(metrics))     # Use a colormap for different colors

    num_categories = len(df)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # Make the plot circular by repeating the first angle at the end
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Prepare the x-ticks: [Topic1, Topic2, ...] with angle wrapping
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"[{i+1}]" for i in range(num_categories)], fontsize=8)

    # Plot each metric as a line
    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        values += values[:1]  # close the loop
        # ax.plot(angles, values, label=metric, marker='o')
        ax.plot(angles, values, label=metric,
        marker=markers[i % len(markers)],
        color=colors(i),
        linewidth=2,
        markersize=6)

    # Add labels to radial axis
    ax.set_ylim(2, 5.15)
    ax.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_yticklabels([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    # Optional: annotate topic names around the circle
    # for i, label in enumerate(df['Number']):
    #     angle = angles[i]
    #     ax.text(angle, 5.4, f"[{i+1}]", ha='center', va='center', fontsize=6)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=8)

    # Save the plot
    plt.tight_layout()
    plt.savefig(fpath.replace('result', 'figure').replace('.json','.png'), dpi=400)
    plt.close()

def compute_result_per_topic(fpath):
    df_tasks_per_topic = pd.read_csv(file_tasks_per_topic) # TOPIC x 3

    with open(fpath, 'r') as fr:
        data = json.load(fr)

    result_df_1 = pd.DataFrame(data["total_conversations"])
    result_df_1['Topic'] = result_df_1['source_tutorial_path'].str.split('/').str[3]
    result_df_2 = pd.DataFrame([d['evaluation'] for d in data["total_conversations"]])[metrics]
    result_df = pd.concat([result_df_1, result_df_2], axis=1)
    group = result_df.groupby('Topic')
    mean_scores = group[metrics].mean() # TOPIC x N_METRIC
    final_df = pd.merge(df_tasks_per_topic, mean_scores, on='Topic', how='inner')
    plot_tasks_per_topic(final_df, fpath)
    final_df.to_csv(fpath.replace('.json', '_topic.csv'), index=False)

if __name__ == "__main__":
    file_tasks_per_topic = 'figure/tasks_per_topic.csv'
    if len(sys.argv)==2:
        """
        Usage example:
        python utils/post_result_per_topic.py result/T-openchat_L-openchat_E-openchat_11297606_corrected.json
        """
        compute_result_per_topic(fpath=sys.argv[1])
    elif len(sys.argv)==1:
        # fpath = "result/T-deepseek-llm_L-deepseek-llm_E-deepseek-llm_012.json"
        fpaths = [
            'T-deepseek-llm_L-deepseek-llm_E-deepseek-llm_11246257_corrected.json',
            'T-qwen2_L-qwen2_E-qwen2_11236934_corrected.json',
            'T-gemma_L-gemma_E-gemma_11269574_corrected.json',
            'T-olmo2_L-olmo2_E-olmo2_11370185_corrected.json',
            'T-openchat_L-openchat_E-openchat_11297606_corrected.json',
            'T-llama3_L-llama3_E-llama3_11232754_corrected.json',
            'T-phi4_L-phi4_E-phi4_11269383_corrected.json',
        ]
        for fpath in fpaths:
            compute_result_per_topic(fpath=f"result/{fpath}")