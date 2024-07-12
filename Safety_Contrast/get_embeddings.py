import pandas as pd
from utils import get_embedding
import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('embedding_model', type=str, help='embedding_model')

args = parser.parse_args()
embedding_model = args.embedding_model

dataset = pd.read_json(f"Safety_Contrast/data/contrast_parsed.jsonl", lines=True)
prompt_types = ['original_prompt', 'prompt_#1','prompt_#2','prompt_#3','prompt_#4','prompt_#5']

embed_dict = {}

for type in prompt_types:
    embed_dict[type] = get_embedding(dataset[type].tolist(), embedding_model)

dir = "Safety_Contrast/embeddings"
if not os.path.exists(dir):
    os.makedirs(dir)


full_path = f"Safety_Contrast/embeddings/embeddings_{embedding_model}.jsonl"
with open(full_path, mode = "w", encoding="utf-8") as wf:
    for data, original_emb, p1_emb,p2_emb,p3_emb,p4_emb,p5_emb in zip(dataset.iterrows(), embed_dict['original_prompt'], 
                                                                      embed_dict['prompt_#1'],embed_dict['prompt_#2'],embed_dict['prompt_#3'],embed_dict['prompt_#4'] , embed_dict['prompt_#5']):
        row = data[1]
        id = row['id']
        category = row['category']

        original_tup = (row['original_prompt'], original_emb)
        p1_tup = (row['prompt_#1'], p1_emb)
        p2_tup = (row['prompt_#2'], p2_emb)
        p3_tup = (row['prompt_#3'], p3_emb)
        p4_tup = (row['prompt_#4'], p4_emb)
        p5_tup = (row['prompt_#5'], p5_emb)

        output = {"id": id, "category": category, "original": original_tup, "p1": p1_tup, "p2": p2_tup,"p3": p3_tup,"p4": p4_tup,"p5": p5_tup}
        output = json.dumps(output)
        wf.write(output + '\n')