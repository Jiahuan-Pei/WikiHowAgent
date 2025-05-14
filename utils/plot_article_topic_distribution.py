import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

def plot_radar_with_topic_bars(df):
    # === Aggregate & Sort ===
    parent_counts = df.groupby('Parent Topic')['Count'].sum().reset_index()
    parent_counts = parent_counts.sort_values(by='Count', ascending=False)
    df['Parent Topic'] = pd.Categorical(df['Parent Topic'], categories=parent_counts['Parent Topic'], ordered=True)

    # === Radar chart setup ===
    N = len(parent_counts)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # === Plot Setup ===
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'polar': True})

    # Plot Parent Topic Totals Outline
    values = parent_counts['Count'].tolist()
    values += values[:1]
    # ax.plot(angles, values, color='red', linewidth=2, linestyle='solid', label='Total Count')
    # ax.fill(angles, values, color='red', alpha=0.1)

    # === Assign Unique Colors to Each Topic ===
    unique_topics = df['Topic'].unique()
    topic_colors = dict(zip(unique_topics, sns.color_palette('pastel', len(unique_topics))))

    # === Plot Bars for Each Topic ===
    for idx, parent in enumerate(parent_counts['Parent Topic']):
        topics = df[df['Parent Topic'] == parent].sort_values(by='Count', ascending=False)
        # We will set the bar for each topic within the parent group
        width = 2 * np.pi / N * 0.7  # Bar width, adjust for overlap
        for i, row in enumerate(topics.iterrows()):
            _, row = row
            angle = angles[idx] + (i * width / len(topics))  # Distribute topics evenly within the parent
            ax.bar(
                angle,
                row['Count'],
                width=width / len(topics),  # Smaller width for individual topics
                color=topic_colors[row['Topic']],
                edgecolor='black',
                alpha=0.9
            )

    # === Y-axis Count Labels ===
    # max_count = parent_counts['Count'].max()
    max_count = df['Count'].max()
    yticks = list(range(0, int(max_count + 50), 50))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=8)

    # === X-axis Parent Topic Labels ===
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(parent_counts['Parent Topic'], fontsize=9, ha='center', rotation=45)

    # === Title and Grid ===
    # ax.set_title('Radar Chart: Topic Counts per Parent Topic (Separate Bars for Each Topic)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # === Optional: Topic Legend ===
    # sampled_topics = df.groupby('Topic')['Count'].sum().sort_values(ascending=False).head(15).index  # Top 15 topics
    # legend_patches = [mpatches.Patch(color=topic_colors[topic], label=topic) for topic in sampled_topics]
    # plt.legend(handles=legend_patches, bbox_to_anchor=(1.2, 1), loc='upper left', fontsize=8, title='Top Topics')

    plt.tight_layout()
    plt.show()

def plot_radar_with_parent_bars(df):
    # === Aggregate & Sort ===
    parent_counts = df.groupby('Parent Topic')['Count'].sum().reset_index()
    parent_counts = parent_counts.sort_values(by='Count', ascending=False)
    df['Parent Topic'] = pd.Categorical(df['Parent Topic'], categories=parent_counts['Parent Topic'], ordered=True)

    # === Radar chart setup ===
    N = len(parent_counts)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

    # Plot Parent Topic Totals Outline
    values = parent_counts['Count'].tolist()
    values += values[:1]
    # ax.plot(angles, values, color='olive', linewidth=2, linestyle='solid', label='Total Count')
    # ax.fill(angles, values, color='olive', alpha=0.1)

    # === Assign better colors per Topic ===
    unique_topics = df['Topic'].unique()

    # Using a colorblind-friendly color palette (cubehelix)
    # topic_colors = dict(zip(unique_topics, sns.cubehelix_palette(len(unique_topics), start=2, rot=0, dark=0.1, light=0.8, reverse=True)))
    topic_colors = dict(zip(unique_topics, sns.cubehelix_palette(len(unique_topics), rot=-.2)))
    # topic_colors = dict(zip(unique_topics, sns.light_palette("#79C", len(unique_topics))))

    # === Plot stacked bars with topic-specific colors ===
    for idx, parent in enumerate(parent_counts['Parent Topic']):
        topics = df[df['Parent Topic'] == parent].sort_values(by='Count', ascending=False)
        current_base = 0
        for _, row in topics.iterrows():
            ax.bar(
                angles[idx],
                row['Count'],
                width=2 * np.pi / N * 0.7,
                bottom=current_base,
                color=topic_colors[row['Topic']],
                edgecolor='black',
                alpha=0.9
            )
            current_base += row['Count']

    # === Y-axis count labels ===
    max_count = parent_counts['Count'].max()
    step = 100
    yticks = list(range(0, int(max_count + step), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=8)
    # Change position of ytick labels (move around the circle)
    ax.set_rlabel_position(-30)  # Adjust this angle to place radial labels where you want
    # Remove outer circle
    # ax.spines['polar'].set_visible(False)

    # === X-axis Parent Topic labels ===
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(parent_counts['Parent Topic'], fontsize=8, ha='center', rotation=45)

    ax.set_xticks([])  # Remove default xtick labels
    ax.spines['polar'].set_visible(True)
    # Manually add labels with padding control
    for i, label in enumerate(parent_counts['Parent Topic']):
        angle = angles[i]
        ax.text(
            angle, 
            1.4 * parent_counts['Count'].max(),  # Adjust 1.05 to control distance from circle
            label, 
            ha='center', 
            va='center', 
            fontsize=8,
            # rotation=np.degrees(angle),  # Optional: Rotate label to follow circle
            # rotation_mode='anchor'
        )
    # === Title & Grid ===
    # ax.set_title('Radar Chart: Topics per Parent Topic (Distinct Colors per Topic)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_sunburst_topics(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    # Sunburst chart
    fig = px.sunburst(df, path=['Parent Topic', 'Topic'], values='Count')
    
    # Adjust Layout for Tight Display
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  # Left, Right, Top, Bottom margins set to zero
        paper_bgcolor='white',             # Optional: clean white background
    )

    fig.show()
    # Save as high-quality PNG or SVG
    fig.write_image("figure/sunburst_chart_high_res.png", scale=5)  # PNG high-res


if __name__ == "__main__":
    # Load your df with columns: Number, Topic, Parent Topic
    df = pd.read_csv('data/tasks_per_topic_categorization.csv')
    # plot_radar_with_topic_bars(df)
    # plot_radar_with_parent_bars(df)
    plot_sunburst_topics(df)