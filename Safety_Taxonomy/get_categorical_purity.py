from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json
import yaml
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('../'))
from utils import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_model', type=str, help='embedding_model')
    parser.add_argument('top_k',type=int,help='top_k similarity',default=10)
    args = parser.parse_args()

    embedding_model = args.embedding_model
    top_k = args.top_k

    df = pd.read_json(f"Safety_Taxonomy/embeddings/embeddings_{embedding_model}.jsonl", lines=True)
    
    dir = "Safety_Taxonomy/CP"
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open('./Safety_Taxonomy/mapping_names.yaml', 'r') as f:
        mapping_data = yaml.safe_load(f)
    mapping = mapping_data['embedding_models']
        
    full_path = f"Safety_Taxonomy/CP/Categorical_Purity_{embedding_model}_top_{top_k}.jsonl" 
    with open(full_path,'w') as wf:
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
            wf.write(json.dumps(result_dict)+'\n')
    
    df = pd.read_json(full_path,lines=True)
    for _, row in df.iterrows():
        print(row['risk_category'], ':' ,round(row["categoric purity"],3))
