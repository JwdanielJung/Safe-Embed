import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml

def tsne_single_visualization(df, embedding_model):
    labels = df[df['embedding_model'] == embedding_model]['risk_category'].tolist()
    # Normalize the embeddings
    scaler = StandardScaler()
    embedding = np.array(df[df['embedding_model'] == embedding_model]['embedding'].tolist())
    embedding_normalized = scaler.fit_transform(embedding)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embedding_normalized)

    with open('mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)
    embedding_model = mapping_data['embedding_models'].get(embedding_model)

    # Plot the result
    plt.figure(figsize=(15, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pd.factorize(labels)[0], cmap='tab10')
    plt.title(f't-SNE of {embedding_model} Embeddings Colored by Class')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True)

    # Add legend
    unique_labels = np.unique(labels)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_labels, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'tsne/tsne_{embedding_model}.png',bbox_inches='tight')
    plt.show()

# Define the palette
palette = [
    (237, 28, 36), (255, 127, 39), (255, 242, 0),
    (181, 230, 29), (34, 177, 76), (153, 217, 234),
    (0, 162, 232), (63, 72, 204), (163, 73, 164),
    (120, 67, 21), (128, 128, 128), (0, 0, 0)
]

# Normalize palette to [0, 1] range for matplotlib
palette = np.array(palette) / 255.0

def tsne_visualization(df, embedding_model):
    labels = df[df['embedding_model'] == embedding_model]['risk_category'].tolist()
    # Normalize the embeddings
    scaler = StandardScaler()
    embedding = np.array(df[df['embedding_model'] == embedding_model]['embedding'].tolist())
    embedding_normalized = scaler.fit_transform(embedding)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embedding_normalized)
    
    return embeddings_2d, labels

def plot_tsne_subplots(df, embedding_models):
    with open('mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)
    
    risk_short_version = mapping_data['risk_short_version']
    sorted_unique_labels = sorted(risk_short_version.values())
    embedding_model_names = mapping_data['embedding_models']
    
    color_mapping = {label: palette[i] for i, label in enumerate(sorted_unique_labels)}

    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 10))
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 18))

    # Manually set the first two embedding models to ST5-XXL and Unsup-SimCSE
    embedding_models = ['st5-xxl', 'unsup-simcse'] + [model for model in embedding_models if model not in ['st5-xxl', 'unsup-simcse']]
    
    # Plot the first two embedding models side by side
    for i, embedding_model in enumerate(embedding_models[:2]):
        embeddings_2d, labels = tsne_visualization(df, embedding_model)
        colors = [color_mapping[label] for label in labels]
        scatter = axes1[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='Pastel1', s=80)  # Increase point size with s parameter
        embedding_model_name = embedding_model_names.get(embedding_model, embedding_model)
        axes1[i].set_title(f't-SNE Visualization for {embedding_model_name}', fontsize=14)
        axes1[i].set_xlabel('t-SNE feature 1', fontsize=12)
        axes1[i].set_ylabel('t-SNE feature 2', fontsize=12)
        axes1[i].grid(True)
    
    # Add a single legend for the first figure
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in range(len(sorted_unique_labels))]
    fig1.legend(handles, sorted_unique_labels, title="Risk Categories", bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize='xx-large', title_fontsize='xx-large', markerscale=1)  # Increase legend size with markerscale
    fig1.tight_layout()
    fig1.savefig('tsne/st5_xxl_unsup_simcse_tsne.png', bbox_inches='tight')
    
    # Plot the remaining embedding models in a 3x3 grid
    for i, embedding_model in enumerate(embedding_models[2:]):
        row, col = divmod(i, 3)
        embeddings_2d, labels = tsne_visualization(df, embedding_model)
        colors = [color_mapping[label] for label in labels]
        scatter = axes2[row, col].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='Pastel1', s=30)  # Adjust point size as desired
        embedding_model_name = embedding_model_names.get(embedding_model, embedding_model)
        axes2[row, col].set_title(f't-SNE Visualization for {embedding_model_name}', fontsize=14)
        axes2[row, col].set_xlabel('t-SNE feature 1', fontsize=12)
        axes2[row, col].set_ylabel('t-SNE feature 2', fontsize=12)
        axes2[row, col].grid(True)
    
    # Remove any unused subplots
    for i in range(len(embedding_models[2:]), 9):
        row, col = divmod(i, 3)
        fig2.delaxes(axes2[row, col])
    
    # Add a single legend for the second figure
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in range(len(sorted_unique_labels))]
    fig2.legend(handles, sorted_unique_labels, title="Risk Categories", bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize='xx-large', title_fontsize='xx-large', markerscale=1)  # Increase legend size with markerscale
    fig2.tight_layout()
    fig2.savefig('tsne/other_models_tsne', bbox_inches='tight')

    plt.show()

def cp_heatmap_visualization(df):
    with open('mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)

    risk_order = list(mapping_data['risk_short_version'].values())
    df['risk_category'] = pd.Categorical(df['risk_category'], categories=risk_order, ordered=True)

    embedding_order = list(mapping_data['embedding_models'].values())
    df['embedding_model'] = pd.Categorical(df['embedding_model'], categories=embedding_order, ordered=True)

    heatmap_data = df.pivot(index='embedding_model', columns='risk_category', values='categoric purity')

    # Create the heatmap with specified adjustments
    plt.figure(figsize=(16, 8))
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap="Blues", cbar=True, vmin=0, vmax=1, annot_kws={'size': 13}, cbar_kws={'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8})

    # Adjust colorbar position
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=10)
    heatmap.collections[0].colorbar.ax.xaxis.set_ticks_position('bottom')
    heatmap.collections[0].colorbar.ax.xaxis.set_label_position('bottom')

    # Remove titles and axis labels
    heatmap.set_title('')
    heatmap.set_xlabel('')
    heatmap.set_ylabel('')
    
    # Adjust category names to wrap into two lines if they are too long
    max_label_length = 10  # You can adjust this value as needed
    xticks_labels = heatmap.get_xticklabels()
    new_labels = [label.get_text() if len(label.get_text()) <= max_label_length else label.get_text().replace(' ', '\n', 2) for label in xticks_labels]
    heatmap.set_xticklabels(new_labels)

    # Adjust labels position
    heatmap.xaxis.set_label_position('top')
    heatmap.xaxis.tick_top()
    plt.xticks(rotation=30, ha='center',fontsize=13)
    plt.yticks(rotation=0, fontsize=13)

    # Add black line
    line_positions = ['Sup-SimCSE', 'ST5-Base', 'text-embedding-3-large', 'LLM2vec-Mistral']
    for pos in line_positions:
        plt.axhline(y=embedding_order.index(pos), color='black', linewidth=2)
    
    # Adjust layout to remove right and bottom whitespace
    plt.tight_layout()
    # plt.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.01)

    plt.savefig('chart/cp_heatmap.png')
    plt.show()

def cp_bar_chart_visualization(df):
    with open('mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)

    embedding_order = list(mapping_data['embedding_models'].values())
    df['embedding_model'] = pd.Categorical(df['embedding_model'], categories=embedding_order, ordered=True)

    # Calculate mean categoric purity for each embedding model
    mean_cp = df.groupby('embedding_model')['categoric purity'].mean().reset_index()

    # Visualization
    plt.figure(figsize=(15, 8))
    barplot = sns.barplot(x='embedding_model', y='categoric purity', data=mean_cp, order=embedding_order, color='green')

    # Set title and labels
    plt.title('Average CP by Sentence Encoders', fontsize=25)
    # plt.xlabel('Embedding Model')
    plt.ylabel('Average CP', fontsize=23)

    # Remove x-axis label
    barplot.set(xlabel=None)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)

    # Set y-axis limits and ticks
    plt.ylim(0.5, 0.8)
    plt.yticks(ticks=np.arange(0.5, 0.85, 0.05))  # Set tick interval to 0.05 starting from 0.5

    # Add data labels
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points',
                         fontsize=20)

    line_positions = ['Sup-SimCSE', 'ST5-Base', 'text-embedding-3-large', 'LLM2vec-Mistral']
    for pos in line_positions:
        plt.axvline(x=embedding_order.index(pos) - 0.5, color='black', linestyle='--', linewidth=3)

    # line_position = embedding_order.index('Sup-SimCSE') - 0.5
    # plt.axvline(x=line_position, color='black', linestyle='--', linewidth=2)
                         
    # Adjust the position of the plot
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9, bottom=0.2)
    

    # Save the plot
    plt.savefig('chart/cp_bar.png')
    plt.show()
