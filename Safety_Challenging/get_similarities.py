
import pandas as pd
from utils import get_similarity, normalize_similarity
import statistics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('embedding_model', type=str, help='embedding_model')
args = parser.parse_args()

embedding_model = args.embedding_model


dataset = pd.read_json(f"Safety_Challenging/embeddings/embeddings_{embedding_model}.jsonl", lines=True)
similarities = {}

for index, row in dataset.iterrows():
    type_id = row['type_id']
    type_name = row['type_name']
    unsafe_prompt = row['unsafe_prompt']
    unsafe_embedding = row['unsafe_embedding']
    safe_prompt = row['safe_prompt']
    safe_embedding = row['safe_embedding']


    original_similarity = get_similarity(unsafe_embedding, safe_embedding)
    normalized_similarity = normalize_similarity(original_similarity, embedding_model)

    if type_name in similarities:
        similarities[type_name].append(normalized_similarity)
    else:
        similarities[type_name] = [normalized_similarity]


for type_name, prompt_similarities in similarities.items():
    print(type_name, round(statistics.fmean(prompt_similarities),3))