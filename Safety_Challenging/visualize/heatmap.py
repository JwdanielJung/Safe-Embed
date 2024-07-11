import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data based on user's input
data = {
    "Homonyms": [0.394, 0.390, 0.532, 0.492, 0.462, 0.403, 0.388, 0.370, 0.424, 0.361, 0.391, 0.419],
    "Figurative\nLanguage": [0.296, 0.290, 0.476, 0.476, 0.395, 0.364, 0.325, 0.298, 0.369, 0.336, 0.329, 0.359],
    "Target": [0.494, 0.475, 0.561, 0.513, 0.499, 0.517, 0.492, 0.442, 0.479, 0.427, 0.447, 0.486],
    "Context": [0.653, 0.620, 0.634, 0.625, 0.641, 0.597, 0.593, 0.562, 0.613, 0.429, 0.546, 0.592],
    "Definitions": [0.534, 0.575, 0.594, 0.616, 0.535, 0.475, 0.471, 0.467, 0.473, 0.436, 0.549, 0.520],
    "Group": [0.514, 0.529, 0.587, 0.527, 0.613, 0.575, 0.524, 0.522, 0.528, 0.432, 0.491, 0.531],
    "Action": [0.385, 0.405, 0.499, 0.430, 0.407, 0.344, 0.321, 0.302, 0.367, 0.325, 0.292, 0.371],
    "History": [0.698, 0.720, 0.736, 0.676, 0.668, 0.585, 0.570, 0.524, 0.599, 0.654, 0.581, 0.637],
    "Privacy\nPublic": [0.396, 0.427, 0.439, 0.428, 0.499, 0.405, 0.368, 0.348, 0.290, 0.275, 0.373, 0.386],
    "Privacy\nFictional": [0.511, 0.558, 0.527, 0.552, 0.585, 0.525, 0.514, 0.508, 0.443, 0.360, 0.449, 0.503]
}

# Embedding models mapping with single line
models = [
    "SBERT-all", "SBERT-paraphrase", "Sup-SimCSE", "Unsup-SimCSE",
    "ST5-Base", "ST5-Large", "ST5-XL", "ST5-XXL", 
    "text-embedding-3-large", "LLM2vec-Mistral", "LLM2vec-Llama3", "Average"
]

# Creating DataFrame
df = pd.DataFrame(data, index=models)

# Creating heatmap
plt.figure(figsize=(16, 8))
ax = sns.heatmap(df, annot=True, fmt=".3f", cmap="Reds", linewidths=.5, linecolor='gray', cbar_kws={"orientation": "horizontal", "pad": 0.02, "shrink": 0.5},annot_kws={"size": 11})

# Add black line between specific rows
for _, spine in ax.spines.items():
    spine.set_visible(True)
ax.hlines([2, 4, 8, 9], *ax.get_xlim(), colors='black', linewidth=2)
ax.hlines([11], *ax.get_xlim(), colors='black', linewidth=4)

# Adjust y-axis to have labels starting from the top without rotation
ax.yaxis.tick_left()
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)

# Adjust x-axis to have labels at the top with rotation to avoid overlapping
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha="center")

# Move color bar to the bottom
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.xaxis.set_ticks_position('bottom')

# Save the heatmap
plt.savefig('XSTest_heatmap.png', bbox_inches='tight')

# Display heatmap
plt.show()
