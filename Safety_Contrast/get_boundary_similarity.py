import pandas as pd
from utils import get_embedding, get_similarity, normalize_similarity
import json
import argparse
import statistics


parser = argparse.ArgumentParser()
parser.add_argument('embedding_model', type=str, help='embedding_model')

args = parser.parse_args()

embedding_model = args.embedding_model

prompt_types = ['original', 'p1','p2','p3','p4','p5']

df = pd.read_json(f"Safety_Contrast/embeddings/embeddings_{embedding_model}.jsonl", lines=True)

maxsim_list = []

for index, row in df.iterrows():
    original_embedding = row[prompt_types[0]][1]
    similarities = []
    for i in range(5):
        similarities.append(get_similarity(original_embedding, row[prompt_types[i+1]][1]))
    
    maxsim = max(similarities)
    normalized_similarity = normalize_similarity(maxsim, embedding_model)
    maxsim_list.append(normalized_similarity)

print("mean:", round(statistics.fmean(maxsim_list),3))

