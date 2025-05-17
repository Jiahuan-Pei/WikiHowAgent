import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

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
    save_path = human_annotation_csv.replace('csv', 'png').replace('data/human_annotation', 'figure').replace('.png', f'_compare_{model_name}.png')
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
    elif tau_axis == 'mean':
        # Compute the mean score and then access the alignment 
        # Kendall significance p-values robust for ties
        human = human_df_filtered[metrics].astype(float).mean(axis=1)
        model = model_df_filtered[metrics].astype(float).mean(axis=1)

        # kappa = cohen_kappa_score(human, model)
        pearson_corr, pearson_p = pearsonr(human, model)
        spearman_corr, spearman_p = spearmanr(human, model)
        kendalltau_corr, kendalltau_p = kendalltau(human, model)

        # print(f"Kappa Scores: {kappa:.3f}")
        print(f"Pearson correlation: {pearson_corr} (p={pearson_p:.4f})")
        print(f"Spearman correlation: {spearman_corr} (p={spearman_p:.4f})")
        print(f"Kendalltau correlation: {kendalltau_corr} (p={kendalltau_p:.4f})")
        return
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

def plot_all_correlation_comparison_over_models(human_annotation_csv, model_evaluation_csv, filenames=None):
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
    save_path = human_annotation_csv.replace('csv', 'compare_by_filename.png').replace('data/human_annotation', 'figure')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()


def plot_metric_correlations_all(human_annotation_csv, model_evaluation_csv):
    # Load data
    human_df = pd.read_csv(human_annotation_csv).drop(columns=['comment'], errors='ignore').fillna(0)
    model_df = pd.read_csv(model_evaluation_csv).fillna(0)

    # Align data: keep only common filenames
    common_filenames = sorted(set(human_df['filename']).intersection(set(model_df['filename'])))
    human_df = human_df[human_df['filename'].isin(common_filenames)]
    model_df = model_df[model_df['filename'].isin(common_filenames)]

    # Drop non-metric columns
    human_metrics = human_df.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore')
    model_metrics = model_df.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore')

    # Compute correlations
    records = []
    for metric in human_metrics.columns:
        try:
            pearson_corr, _ = pearsonr(human_metrics[metric], model_metrics[metric])
        except Exception:
            pearson_corr = 0
        try:
            spearman_corr, _ = spearmanr(human_metrics[metric], model_metrics[metric])
        except Exception:
            spearman_corr = 0
        try:
            kendall_corr, _ = kendalltau(human_metrics[metric], model_metrics[metric])
        except Exception:
            kendall_corr = 0

        records.append({'Metric': metric, 'Correlation': pearson_corr, 'Type': 'Pearson'})
        records.append({'Metric': metric, 'Correlation': spearman_corr, 'Type': 'Spearman'})
        records.append({'Metric': metric, 'Correlation': kendall_corr, 'Type': 'Kendall Tau'})

    corr_df = pd.DataFrame(records)

    # Plot
    plt.figure(figsize=(8, len(human_metrics.columns) * 0.5))
    sns.barplot(data=corr_df, x='Correlation', y='Metric', hue='Type', palette='Set2')
    plt.xlim(-1, 1)
    plt.title('Model-Human Correlations per Metric')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    # Save
    save_path = human_annotation_csv.replace('csv', 'correlation_alltypes_avg.png').replace('data/human_annotation', 'figure')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

def plot_metric_correlations_all_with_significance(human_annotation_csv, model_evaluation_csv):
    hatches = ['xxxxxx', '/////', '']  # For Pearson, Spearman, Kendalltau
    # Load data
    human_df = pd.read_csv(human_annotation_csv).drop(columns=['comment'], errors='ignore').fillna(0)
    model_df = pd.read_csv(model_evaluation_csv).fillna(0)

    # Align data
    common_filenames = sorted(set(human_df['filename']).intersection(set(model_df['filename'])))
    human_df = human_df[human_df['filename'].isin(common_filenames)]
    model_df = model_df[model_df['filename'].isin(common_filenames)]

    # Drop non-metric columns
    human_metrics = human_df.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore')
    model_metrics = model_df.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore')

    # Compute correlations with p-values
    records = []
    for metric in human_metrics.columns:
        # Pearson
        try:
            pearson_corr, pearson_p = pearsonr(human_metrics[metric], model_metrics[metric])
        except Exception:
            pearson_corr, pearson_p = 0, 1.0
        
        # Spearman
        try:
            spearman_corr, spearman_p = spearmanr(human_metrics[metric], model_metrics[metric])
        except Exception:
            spearman_corr, spearman_p = 0, 1.0
        
        # Kendall Tau
        try:
            kendall_corr, kendall_p = kendalltau(human_metrics[metric], model_metrics[metric])
        except Exception:
            kendall_corr, kendall_p = 0, 1.0
        
        # Record values & significance markers
        records.append({'Metric': metric, 'Correlation': pearson_corr, 'p': pearson_p, 'Type': 'Pearson'})
        records.append({'Metric': metric, 'Correlation': spearman_corr, 'p': spearman_p, 'Type': 'Spearman'})
        records.append({'Metric': metric, 'Correlation': kendall_corr, 'p': kendall_p, 'Type': 'Kendall'})

    corr_df = pd.DataFrame(records)

    # Add significance markers
    def significance_marker(p):
        if p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    corr_df['Significance'] = corr_df['p'].apply(significance_marker)

    # Plot
    # plt.figure(figsize=(4, len(human_metrics.columns) * 0.3))
    
    plt.figure(figsize=(4, 1.8))
    bars = sns.barplot(data=corr_df, x='Correlation', y='Metric', hue='Type', palette='pastel')

    # Annotate p-value significance on bars
    for i, row in corr_df.iterrows():
        plt.text(row['Correlation'] + 0.05 * (1 if row['Correlation'] >= 0 else -1) -0.025,  # offset a bit from the bar
                 i // 3 + (0.25 * (['Pearson', 'Spearman', 'Kendall'].index(row['Type']) - 1)+0.15),  # adjust y pos
                 row['Significance'], 
                 color='black', fontsize=8, ha='center', va='center')

    # Apply hatch patterns to each correlation type
    for i, bar_container in enumerate(bars.containers):
        for patch in bar_container.patches:
            patch.set_hatch(hatches[i])

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlim(-0.45, 0.45)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    # plt.title('Model-Human Correlations per Metric (with significance)')
    plt.ylabel('')
    plt.xlabel('')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=5)
    plt.tight_layout()

    # Save figure
    save_path = human_annotation_csv.replace('csv', 'correlation_alltypes_sig.png').replace('data/human_annotation', 'figure')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()


def human_alignment(human_annotation_csv, model_evaluation_csv, tau_axis):
    if tau_axis == 'mean':
        for f in json_files:
            print('*'*50, f)
            measure_correlation(human_annotation_csv=human_annotation_csv, model_evaluation_csv=model_evaluation_csv, filter_filename=f, tau_axis=tau_axis)
        return
    # Per model
    for f in json_files:
        print('*'*50, f)
        tau_values = measure_correlation(human_annotation_csv=human_annotation_csv, model_evaluation_csv=model_evaluation_csv, filter_filename=f, tau_axis=tau_axis)
        model_name = f.split('_')[0].split('-')[1]
        caculate_alignment(tau_values, model_name=model_name,tau_axis=tau_axis)
    # Overall
    tau_values = measure_correlation(human_annotation_csv=human_annotation_csv, model_evaluation_csv=model_evaluation_csv, filter_filename=None, tau_axis=tau_axis)
    caculate_alignment(tau_values, model_name='all',tau_axis=tau_axis)

def analysis_human_annotator(human_annotation_csv, model_evaluation_csv):
    # 1. Calculate alignment over sample and visulize it
    # human_alignment(human_annotation_csv, model_evaluation_csv, tau_axis='sample')
    # 2. Calculate alignment over metric and visulize it
    # human_alignment(human_annotation_csv, model_evaluation_csv, tau_axis='metric')
    # 3. Calculate alignment over mean metric
    # human_alignment(human_annotation_csv, model_evaluation_csv, tau_axis='mean')
    # 3. Plot overall comparsion between models and human
    plot_all_correlation_comparison_over_models(human_annotation_csv, model_evaluation_csv, filenames=json_files)    

def compare_kappa_annotations_by_index(file1, file2, annotation_columns, output_file):
    """
    Compare Cohen's Kappa for multiple annotation dimensions between two CSV files by index (row number).

    Args:
        file1 (str): Path to the first CSV file (annotator 1).
        file2 (str): Path to the second CSV file (annotator 2).
        annotation_columns (list): List of annotation columns to compare.

    Returns:
        dict: Dictionary of kappa values per annotation dimension.
    """

    # Read CSV files (safe load, ignore comments)
    with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
        df1 = pd.read_csv(f1).fillna(0)

    with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
        df2 = pd.read_csv(f2).fillna(0)

    # Plotting two subplots: Human vs Model
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))  # 1 row, 2 columns

    # Subplot 1: Human annotation 1
    sns.boxplot(data=df1.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore'), 
                orient='h', palette='pastel', ax=axes[0])
    axes[0].set_title('Human Annotation 1')
    # axes[0].set_xlim(0.8, 5.2)

    # Subplot 2: Human annotation 2
    sns.boxplot(data=df2.drop(columns=['filename', 'conversation_id', 'title'], errors='ignore'), 
                orient='h', palette='pastel', ax=axes[1])

    axes[1].set_title(f'Human Annotation 2')
    axes[1].set_yticklabels([]) # Remove the tick labels for the second subfigure
    # axes[1].set_xlim(0.8, 5.2)

    plt.tight_layout()
    
    # Save figure with consistent naming
    save_path = output_file.replace('csv', 'png').replace('data/human_annotation', 'figure').replace('.png', f'_compare_annotators.png')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # plt.show()

    # Ensure same number of rows, warn if not
    if len(df1) != len(df2):
        print(f"Warning: CSV files have different number of rows! df1: {len(df1)}, df2: {len(df2)}")

    # Slice annotation columns directly (aligned by index)
    kappa_results = {}
    for col in annotation_columns:
        print('-'*50, f'{col}')
        y_true = df1[col]
        y_pred = df2[col]
        kappa = cohen_kappa_score(y_true, y_pred)
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        kendalltau_corr, kendalltau_p = kendalltau(y_true, y_pred)
        kappa_results[col] = kappa
        print(f"Kappa Scores: {kappa:.3f}")
        print(f"Pearson correlation: {pearson_corr} (p={pearson_p:.4f})")
        print(f"Spearman correlation: {spearman_corr} (p={spearman_p:.4f})")
        print(f"Kendalltau correlation: {kendalltau_corr} (p={kendalltau_p:.4f})")

    # Print results
    # print("Kappa Scores per Dimension:")
    # for dim, score in kappa_results.items():
    #     print(f"{dim}: {score:.3f}")

    mean_score_list_1 = df1[metrics].mean(axis=1)
    mean_score_list_2 = df2[metrics].mean(axis=1)
    # mean_kappa = cohen_kappa_score(mean_score_list_1, mean_score_list_2)
    # Use Weighted Kappa for Ordinal Data
    # Discretize (round to nearest integer)
    discrete_list_1 = np.round(mean_score_list_1).astype(int)
    discrete_list_2 = np.round(mean_score_list_2).astype(int)
    mean_kappa = cohen_kappa_score(discrete_list_1, discrete_list_2, weights='quadratic')  # 'quadratic' or 'linear' 
    pearson_corr, pearson_p = pearsonr(mean_score_list_1, mean_score_list_2)
    spearman_corr, spearman_p = spearmanr(mean_score_list_1, mean_score_list_2)
    kendalltau_corr, kendalltau_p = kendalltau(mean_score_list_1, mean_score_list_2)
    print('='*50)
    print(f"Overall Kappa Scores: {mean_kappa}")
    print(f"Overall Pearson correlation: {pearson_corr} (p={pearson_p:.4f})")
    print(f"Overall Spearman correlation: {spearman_corr} (p={spearman_p:.4f})")
    print(f"Overall Kendalltau correlation: {kendalltau_corr} (p={kendalltau_p:.4f})")

    # Compute averages and replace in df1
    for col in annotation_columns:
        df1[col] = (df1[col] + df2[col]) / 2

    # Save result (overwriting file1 name with _merged)
    df1.to_csv(output_file, index=False)
    print(f"Averaged scores saved to {output_file}")

    return kappa_results


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
    # Step 1: Compute kappa value between human annotators and merge them
    human_annotation_csv_1='data/human_annotation/p25_Yifan.csv'
    human_annotation_csv_2='data/human_annotation/p25_Yue.csv'
    human_annotation_csv_merged = human_annotation_csv_1.replace('.csv', '_merged.csv')
    model_evaluation_csv='data/human_annotation/human_eval_conversation_mono_p25.csv'

    # Compare the agreement of the two annotators
    # compare_kappa_annotations_by_index(file1=human_annotation_csv_1, file2=human_annotation_csv_2, annotation_columns=metrics, output_file=human_annotation_csv_merged)
    
    # Step 2: Human alignment analysis
    # analysis_human_annotator(human_annotation_csv=human_annotation_csv_merged, model_evaluation_csv=model_evaluation_csv)
    plot_metric_correlations_all_with_significance(human_annotation_csv=human_annotation_csv_merged, model_evaluation_csv=model_evaluation_csv)