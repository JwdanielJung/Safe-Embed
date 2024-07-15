import argparse
import json
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('../'))
from utils import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_model', type=str, help='embedding_model')
    args = parser.parse_args()

    embedding_model = args.embedding_model

    df = pd.read_csv('./dataset/do_not_answer.csv')
    with open('./Safety_Taxonomy/mapping_names.yaml', 'r') as file:
        risk_short_version = yaml.safe_load(file)['risk_short_version']
    df['types_of_harm'] = df['types_of_harm'].map(risk_short_version)
    dataset = df[['types_of_harm','question']]

    query = dataset['question'].tolist()
    risk_category = dataset['types_of_harm'].tolist()

    embedding = get_embedding(query, embedding_model)

    dir = "Safety_Taxonomy/embeddings"
    if not os.path.exists(dir):
        os.makedirs(dir)

    full_path = f"Safety_Taxonomy/embeddings/embeddings_{embedding_model}.jsonl"
    with open(full_path, mode = "w", encoding="utf-8") as wf:
        for q, risk, embed in zip(query, risk_category, embedding):
            results = {
                    'query': q,
                    'risk_category': risk,
                    'embedding_model': embedding_model,
                    'embedding': embed
                }
            wf.write(json.dumps(results) + '\n')
