import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

palette = sns.color_palette("pastel")  # Try "Set2", "pastel", "husl", etc.
hatches = ['///','\\\\\\', '']  # For Pearson, Spearman, Kendalltau

# Data preparation
data = {
    'Model': ['deepseek', 'qwen2', 'gemma', 'olmo2', 'openchat', 'llama3', 'phi4'],
    'Pearson': [-0.339, 0.362, 0.263, 0.390, 0.082, -0.058, 0.052],
    'Spearman': [-0.358, 0.387, 0.270, 0.188, 0.128, -0.137, 0.092],
    'Kendalltau': [-0.318, 0.325, 0.251, 0.158, 0.108, -0.090, 0.070]
}

df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# P-values data
p_values = {
    'Model': ['deepseek', 'qwen2', 'gemma', 'olmo2', 'openchat', 'llama3', 'phi4'],
    'Pearson_p': [0.0973, 0.0756, 0.2041, 0.0537, 0.6971, 0.7843, 0.8065],
    'Spearman_p': [0.0785, 0.0562, 0.1919, 0.3679, 0.5433, 0.5127, 0.6626],
    'Kendalltau_p': [0.0585, 0.0506, 0.1124, 0.3568, 0.5319, 0.5772, 0.6827]
}

p_df = pd.DataFrame(p_values)
p_df.set_index('Model', inplace=True)

# Plot with annotations
fig, ax = plt.subplots(figsize=(4, 2.5))
bars = df.plot(kind='barh', ax=ax, color=palette, width=0.75)

# Apply hatch patterns to each correlation type
for i, bar_container in enumerate(bars.containers):
    for patch in bar_container.patches:
        patch.set_hatch(hatches[i])

# Add significance markers for horizontal bars
for i, model in enumerate(df.index):
    for j, corr_type in enumerate(df.columns):
        p_val = p_df.loc[model, f"{corr_type}_p"]
        p_val = round(p_val, 2)
        width = df.loc[model, corr_type]
        if p_val < 0.01:
            marker = '**'
        elif p_val < 0.05:
            marker = '*'
        else:
            marker = ''
        if marker:
            offset = 0.15 * (j - 1)  # Adjust spacing between correlation types
            ax.text(width + np.sign(width) * 0.01, i + offset, marker,
                    va='center', ha='left' if width > 0 else 'right', fontsize=8, color='black')


plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
# plt.title('Human Alignment Correlation Results Across Models (Significance Marked)')
plt.tight_layout()
plt.ylabel('')
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
# plt.show()
plt.savefig('figure/human_alignment_per_model.png', dpi=400)