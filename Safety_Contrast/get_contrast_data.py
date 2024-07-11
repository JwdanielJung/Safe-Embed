import os
from utils import get_5_contrasts
import pandas as pd
from tqdm import tqdm
import json
from openai import OpenAI

file_path = 'dataset/do_not_answer.csv'

df = pd.read_csv(file_path)

full_path = "Safety_Contrast/data/contrast_origin.jsonl"

client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

with open(full_path, mode = "w", encoding="utf-8") as wf:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        category = row['types_of_harm']
        question = row['question']
        id = row['id']
        contrasts = get_5_contrasts(question, client, 1, category)
        result = {"id": id, "category": category, "question": question, "contrasts": contrasts}
        result = json.dumps(result)
        wf.write(str(result) + "\n")