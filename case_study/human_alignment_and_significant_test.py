import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from config import metrics
except:
    metrics = ["Clarity", "Engagement", "Coherence", "Depth", "Relevance", "Progress", "Naturalness", "Truthfulness"]
    metrics = [m.lower() for m in metrics]

# models = ['deepseek', 'qwen2', 'gemma', 'olmo2', 'openchat', 'llama3', 'phi4']

def measure_correlation(human_annotation_csv, model_evaluation_csv, filter_filename, tau_axis):
    # Human annotations
    human_df = pd.read_csv(human_annotation_csv)
    human_df = human_df.drop(columns=['comment'], errors='ignore')
    human_df = human_df.fillna(0)
    human_df_filtered = human_df.loc[human_df['filename'] == filter_filename] if filter_filename else human_df

    # Model evaluations
    model_df = pd.read_csv(model_evaluation_csv)
    model_df = model_df.fillna(0)
    model_df_filtered = model_df.loc[model_df['filename'] == filter_filename] if filter_filename else model_df

    # Plotting two subplots: Human vs Model
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))  # 1 row, 2 columns

    # Subplot 1: Human annotations
    sns.boxplot(data=human_df_filtered.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore'), 
                orient='h', palette='pastel', ax=axes[0])
    axes[0].set_title('Human Annotations')
    # axes[0].set_xlim(0.8, 5.2)

    # Subplot 2: Model evaluations
    sns.boxplot(data=model_df_filtered.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore'), 
                orient='h', palette='pastel', ax=axes[1])
    model_name = filter_filename.split('_')[0].split('-')[1] if filter_filename else 'all'
    axes[1].set_title(f'Model-{model_name} Evaluations')
    axes[1].set_yticklabels([]) # Remove the tick labels for the second subfigure
    # axes[1].set_xlim(0.8, 5.2)

    plt.tight_layout()
    
    # Save figure with consistent naming
    save_path = human_annotation_csv.replace('csv', 'png').replace('case_study', 'figure').replace('.png', f'_compare_{model_name}.png')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()

    tau_values = []
    if tau_axis == 'sample':
        for i in range(len(human_df_filtered)):
            h = human_df_filtered.iloc[i][metrics].astype(float)
            m = model_df_filtered.iloc[i][metrics].astype(float)
            kendall_corr, kendall_p = kendalltau(h, m)
            if not np.isnan(kendall_corr):
                tau_values.append(kendall_corr)
                print(f"Sample {i} - Kendall: {kendall_corr:.3f} (p={kendall_p:.3f})")
        print(f"Valid ratio: {len(tau_values)/len(human_df_filtered)*100:.2f}")
    elif tau_axis == 'metric':
        for metric in metrics:
            metric = metric.lower()
            # Kendall significance p-values robust for ties
            kendall_corr, kendall_p = kendalltau(human_df_filtered[f'{metric}'].astype(float), model_df_filtered[f'{metric}'].astype(float))
            if not np.isnan(kendall_corr):
                tau_values.append(kendall_corr)
                print(f"{metric} - Kendall: {kendall_corr:.3f} (p={kendall_p:.3f})")
    return np.array(tau_values)   


def caculate_alignment(tau_values, model_name=None, tau_axis=None):
    # your real Kendall's tau per sample
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import kendalltau, ttest_1samp, wilcoxon
    # Example: synthetic tau values for 100 samples (replace with real values)
    # tau_values = np.random.uniform(0.3, 0.8, size=100)  # your real Kendall's tau per sample

    # Aggregate metrics
    mean_tau = np.mean(tau_values)
    std_tau = np.std(tau_values, ddof=1)
    se_tau = std_tau / np.sqrt(len(tau_values))

    # 95% Confidence Interval
    ci_lower = mean_tau - 1.96 * se_tau
    ci_upper = mean_tau + 1.96 * se_tau

    print(f"Mean Kendall's Tau: {mean_tau:.3f} ± {1.96 * se_tau:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

    # Hypothesis Testing (is mean_tau > 0?)
    t_stat, p_value_ttest = ttest_1samp(tau_values, 0)
    w_stat, p_value_wilcoxon = wilcoxon(tau_values - 0)  # two-sided test

    print(f"T-test p-value (H0: mean_tau = 0): {p_value_ttest:.3e}")
    print(f"Wilcoxon signed-rank p-value (H0: median_tau = 0): {p_value_wilcoxon:.3e}")

    # Visualization
    plt.figure(figsize=(12, 5))

    # Histogram + KDE
    sns.histplot(tau_values, kde=True, bins=20, color='lightgreen')
    plt.axvline(mean_tau, color='red', linestyle='--', label=f'Mean = {mean_tau:.2f}')
    plt.axvspan(ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
    plt.xlabel(f"Kendall's Tau per {tau_axis}")
    plt.ylabel('Frequency')
    title1 = f"Distribution of Kendall's Tau (Model-{model_name} vs Human)"
    plt.title(title1)
    plt.legend()
    plt.savefig(f'figure/{title1}.png', dpi=500, bbox_inches='tight')
    # plt.show()

    # Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=tau_values, color='lightblue')
    title2 = f"Kendall's Tau per {tau_axis} (Boxplot)"
    plt.title(title2)
    plt.xlabel("Kendall's Tau")
    # plt.show()
    plt.savefig(f'figure/{title2}.png', dpi=500, bbox_inches='tight')

def plot_all_comparison_by_filenames(human_annotation_csv, model_evaluation_csv, filenames=None):
    # Load data
    human_df = pd.read_csv(human_annotation_csv).drop(columns=['comment'], errors='ignore').fillna(0)
    model_df = pd.read_csv(model_evaluation_csv).fillna(0)

    if filenames is None:
        filenames = sorted(set(human_df['filename']).intersection(set(model_df['filename'])))

    n_files = len(filenames)
    n_cols = 8
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1, n_rows * 3))
    axes = axes.flatten()

    for idx, filename in enumerate(filenames):
        ax = axes[idx]

        # Filter data for this filename
        human_sample = human_df[human_df['filename'] == filename].drop(columns=['filename', 'conversation_id', 'title'], errors='ignore')
        model_sample = model_df[model_df['filename'] == filename].drop(columns=['filename', 'conversation_id', 'title'], errors='ignore')

        # Melt to long-form
        human_melt = human_sample.melt(var_name='Metric', value_name='Score')
        human_melt['Source'] = 'Human'

        model_melt = model_sample.melt(var_name='Metric', value_name='Score')
        model_melt['Source'] = 'Model'

        combined_df = pd.concat([human_melt, model_melt], ignore_index=True)

        # Draw boxplot
        # sns.boxplot(data=combined_df, x='Score', y='Metric', hue='Source', palette='colorblind', ax=ax, fliersize=5, fill=False,  width=0.5)
        sns.violinplot(data=combined_df, x='Score', y='Metric', hue='Source', ax=ax, split=True, inner="quart", fill=False, palette={"Model": "red", "Human": ".35"}) # , palette='pastel', , bw_adjust=0.5, inner="quart"

        if idx % n_cols > 0:
            ax.set_yticklabels([]) # Remove the tick labels
        # if idx // n_cols == 0:
        #     ax.set_xticklabels([]) # Remove the tick labels
        # else:
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.1, 6)
        ax.set_ylabel('')  # Remove the y-axis label ('Metric')
        ax.set_xlabel('')  # Remove the y-axis label ('Score')
        ax.set_title(f'{filename.split("_")[0].split("-")[1]}', fontsize=10)
        # ax.legend(loc='lower right', fontsize=8, frameon=True)
                # Don't add legend to individual plots
        ax.legend_.remove()

        # Store handles and labels for a common legend
        if idx == 0:
            # Get the first plot's handles and labels
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    # Add a single legend for the entire figure (outside the subplots)
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, framealpha=1.0) # , frameon=True

    # Remove empty axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = human_annotation_csv.replace('csv', 'compare_by_filename.png').replace('case_study', 'figure')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()

def human_alignment(tau_axis):
    # Per model
    for f in json_files:
        print('*'*50, f)
        tau_values = measure_correlation(human_annotation_csv=human_annotation_csv, model_evaluation_csv=model_evaluation_csv, filter_filename=f, tau_axis=tau_axis)
        model_name = f.split('_')[0].split('-')[1]
        caculate_alignment(tau_values, model_name=model_name,tau_axis=tau_axis)
    # Overall
    tau_values = measure_correlation(human_annotation_csv=human_annotation_csv, model_evaluation_csv=model_evaluation_csv, filter_filename=None, tau_axis=tau_axis)
    caculate_alignment(tau_values, model_name='all',tau_axis=tau_axis)


if __name__ == "__main__":
    # print('='*100)
    json_files = [
        'T-deepseek-llm_L-deepseek-llm_E-deepseek-llm_11246257_corrected.json',
        'T-qwen2_L-qwen2_E-qwen2_11236934_corrected.json',
        'T-gemma_L-gemma_E-gemma_11269574_corrected.json',
        'T-olmo2_L-olmo2_E-olmo2_11370185_corrected.json',
        'T-openchat_L-openchat_E-openchat_11297606_corrected.json',
        'T-llama3_L-llama3_E-llama3_11232754_corrected.json',
        'T-phi4_L-phi4_E-phi4_11269383_corrected.json',
    ]

    human_annotation_csv='case_study/p25_Yifan.csv'
    model_evaluation_csv='case_study/human_eval_conversation_mono_p25.csv'
    # 1. Calculate alignment over sample and visulize it
    human_alignment(tau_axis='sample')
    # 2. Calculate alignment over metric and visulize it
    human_alignment(tau_axis='metric')
    # 3. Plot overall comparsion between models and human
    # plot_all_comparison_by_filenames(human_annotation_csv='case_study/p25_Yifan.csv',  
    #                             model_evaluation_csv='case_study/human_eval_conversation_mono_p25.csv',
    #                             filenames=json_files)