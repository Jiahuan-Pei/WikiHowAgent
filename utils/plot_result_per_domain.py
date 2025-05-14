import sys, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

metrics = ["Clarity", "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness"]

def plot_tasks_per_domain(df, fpath):
    markers = ['o', 's', 'X', 'D', 'P', '*', '^', 'v', '<', '>']  # Circle, square, diamond, triangle, etc.
    colors = plt.cm.get_cmap('tab10', len(metrics))     # Use a colormap for different colors

    num_categories = len(df)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # Make the plot circular by repeating the first angle at the end
    angles += angles[:1]

    if plot_legend:
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    else:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Prepare the x-ticks: [domain1, domain2, ...] with angle wrapping
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
        linewidth=1,
        markersize=6)
        # Fill area under the plot
        # ax.fill(angles, values, color=colors(i), alpha=0.1)

    # Add labels to radial axis
    ax.set_ylim(2, 5.15)
    ax.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_yticklabels([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    # Optional: annotate domain names around the circle
    # for i, label in enumerate(df['Number']):
    #     angle = angles[i]
    #     ax.text(angle, 5.4, f"[{i+1}]", ha='center', va='center', fontsize=6)

    # Add legend
    if plot_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=10, ncols=len(metrics), framealpha=1.0)
        suffix = '_legend'
    else:
        suffix = ''

    # Save the plot
    plt.tight_layout()
    plt.savefig(fpath.replace('result', 'figure').replace('.json',f'{suffix}.png'), dpi=400)
    plt.close()


def compute_result_per_domain(fpath):
    # Load the Domain-Topic mapping CSV
    df_tasks_per_domain = pd.read_csv(file_tasks_per_domain)  # Columns: Number, Domain, Topic, Count

    # Load experiment results from JSON
    with open(fpath, 'r') as fr:
        data = json.load(fr)

    # Create DataFrame from JSON 'total_conversations'
    result_df_1 = pd.DataFrame(data["total_conversations"])
    result_df_1['Topic'] = result_df_1['source_tutorial_path'].str.split('/').str[3]

    # Extract evaluation metrics (adjust metrics list to your case)
    # metrics = ['accuracy', 'f1', 'precision', 'recall']  # <-- adjust as needed
    result_df_2 = pd.DataFrame([d['evaluation'] for d in data["total_conversations"]])[metrics]

    # Combine topic and metric columns
    result_df = pd.concat([result_df_1[['Topic']], result_df_2], axis=1)

    # Merge with Domain info to get Domain column
    result_with_domain = pd.merge(result_df, df_tasks_per_domain[['Domain', 'Topic']], on='Topic', how='inner')

    # Group by Domain and compute mean scores
    domain_scores = result_with_domain.groupby('Domain')[metrics].mean().reset_index()

    # Optional: Plotting function (if you have one)
    plot_tasks_per_domain(domain_scores, fpath)

    # Save final domain-level scores to CSV
    domain_scores.to_csv(fpath.replace('.json', '_domain.csv'), index=False)

if __name__ == "__main__":
    file_tasks_per_domain = 'data/tasks_per_topic_domain.csv' 
    if len(sys.argv)==2:
        """
        Usage example:
        python utils/plot_result_per_domain.py result/T-openchat_L-openchat_E-openchat_11297606_corrected.json
        """
        plot_legend = True
        compute_result_per_domain(fpath=sys.argv[1])
    elif len(sys.argv)==1:
        plot_legend = False
        fpaths = [
            'T-openchat_L-openchat_E-openchat_11297606_corrected.json',
            'T-deepseek-llm_L-deepseek-llm_E-deepseek-llm_11246257_corrected.json',
            'T-qwen2_L-qwen2_E-qwen2_11236934_corrected.json',
            'T-gemma_L-gemma_E-gemma_11269574_corrected.json',
            'T-olmo2_L-olmo2_E-olmo2_11370185_corrected.json',
            'T-llama3_L-llama3_E-llama3_11232754_corrected.json',
            'T-phi4_L-phi4_E-phi4_11269383_corrected.json',
        ]

        for i, fpath in enumerate(fpaths):
            compute_result_per_domain(fpath=f"result/{fpath}")