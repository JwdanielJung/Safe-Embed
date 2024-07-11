import pandas as pd
from utils import get_embedding
import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('embedding_model', type=str, help='embedding_model')
args = parser.parse_args()

embedding_model = args.embedding_model



dataset = pd.read_csv("Safety_Challenging/xstest_aug.csv")

unsafe_list = dataset['unsafe_prompt'].tolist()
safe_list = dataset['safe_prompt'].tolist()

unsafe_embedding_list = get_embedding(unsafe_list, embedding_model)
safe_embedding_list = get_embedding(safe_list, embedding_model)

dir = "Safety_Challenging/embeddings"
if not os.path.exists(dir):
    os.makedirs(dir)


full_path = f"Safety_Challenging/embeddings/embeddings_{embedding_model}.jsonl"
with open(full_path, mode = "w", encoding="utf-8") as wf:
    for data, unsafe_embedding, safe_embedding in zip(dataset.iterrows(), unsafe_embedding_list, safe_embedding_list):
        row = data[1]
        type_id = row['type_id']
        type_name = row['type']
        unsafe_prompt = row['unsafe_prompt']
        safe_prompt = row['safe_prompt']
        output = {"type_id": type_id, "type_name": type_name, "unsafe_prompt": unsafe_prompt, "unsafe_embedding": unsafe_embedding, "safe_prompt": safe_prompt, "safe_embedding": safe_embedding}
        output = json.dumps(output)
        wf.write(output + '\n')


