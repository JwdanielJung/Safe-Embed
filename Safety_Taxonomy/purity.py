from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import yaml
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('../'))
from utils import *


def load_data(path):
    df = pd.read_csv(path)
    with open('mapping_names.yaml', 'r') as file:
        risk_short_version = yaml.safe_load(file)['risk_short_version']
    df['types_of_harm'] = df['types_of_harm'].map(risk_short_version)
    df = df[['types_of_harm','question']]
    return df

def encoding(query, risk_category, embedding_models):
    with open('embeddings.jsonl','w') as f:
        for embedding_model in tqdm(embedding_models):
            embedding = get_embedding(query, embedding_model)
            for q, risk, embed in zip(query, risk_category, embedding):
                results = {
                    'query': q,
                    'risk_category': risk,
                    'embedding_model': embedding_model,
                    'embedding': embed
                }
                f.write(json.dumps(results) + '\n')


def get_category_purity(df, embedding_models, top_k=10):

    with open('mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)
    mapping = mapping_data['embedding_models']

    with open('cp.jsonl','w') as f:
        for embedding_model in tqdm(embedding_models):
            categories = df[df['embedding_model']==embedding_model]['risk_category'].tolist()
            category_results = defaultdict(list)
            embedding = np.array(df[df['embedding_model']==embedding_model]['embedding'].tolist())
            similarity_matrix = cosine_similarity(embedding)
        
            for i, category in (enumerate(categories)):
                similarities = similarity_matrix[i, :]
                # num_cats = sum([1 for idx, cat in enumerate(categories) if cat == categories[i]])

                top_k_idx = similarities.argsort()[::-1][1:top_k + 1] # exclude self
                results = [int(categories[i] == categories[idx]) for idx in top_k_idx]
                category_results[categories[i]].append(sum(results)/len(results))
            
            result_dict = {}
            for category, result in category_results.items(): 
                categoric_purity = sum(result)/len(result)
                result_dict = {'embedding_model':mapping.get(embedding_model,embedding_model),'risk_category':category,'categoric purity':round(categoric_purity,3),'total':int(len(result)), 'hit@cat':int(sum(result))}
                f.write(json.dumps(result_dict)+'\n')

    return pd.read_json('cp.jsonl', lines=True)