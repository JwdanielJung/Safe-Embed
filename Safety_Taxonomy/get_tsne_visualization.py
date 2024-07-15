from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import argparse
import os

# Define the palette
palette = [
    (237, 28, 36), (255, 127, 39), (255, 242, 0),
    (181, 230, 29), (34, 177, 76), (153, 217, 234),
    (0, 162, 232), (63, 72, 204), (163, 73, 164),
    (120, 67, 21), (128, 128, 128), (0, 0, 0)
]

# Normalize palette to [0, 1] range for matplotlib
palette = np.array(palette) / 255.0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_model', type=str, help='embedding_model')
    parser.add_argument('top_k',type=int,help='top_k similarity',default=10)
    args = parser.parse_args()

    embedding_model = args.embedding_model
    top_k = args.top_k

    df = pd.read_json(f"Safety_Taxonomy/embeddings/embeddings_{embedding_model}.jsonl", lines=True)

    dir = "Safety_Taxonomy/tsne"
    if not os.path.exists(dir):
        os.makedirs(dir)

    labels = df[df['embedding_model'] == embedding_model]['risk_category'].tolist()
    
    # Normalize the embeddings
    scaler = StandardScaler()
    embedding = np.array(df[df['embedding_model'] == embedding_model]['embedding'].tolist())
    embedding_normalized = scaler.fit_transform(embedding)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embedding_normalized)

    # Using official model name for plotting
    with open('./Safety_Taxonomy/mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)
    embedding_model_name = mapping_data['embedding_models'].get(embedding_model)

    # Convert labels to pandas Series
    labels_series = pd.Series(labels)
    label_indices = pd.factorize(labels_series)[0]

    # Create a color map based on the palette
    colors = [palette[i % len(palette)] for i in label_indices]

    # Plot the result
    plt.figure(figsize=(15, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)
    plt.title(f't-SNE visualization of {embedding_model_name}')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True)

    # Add legend
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i % len(palette)], markersize=10) for i in range(len(unique_labels))]
    plt.legend(handles, unique_labels, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./Safety_Taxonomy/tsne/tsne_{embedding_model_name}_top_{top_k}.png', bbox_inches='tight')
    plt.show()
