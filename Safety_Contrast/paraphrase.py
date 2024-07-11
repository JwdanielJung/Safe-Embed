import sys
import os
from utils import get_5_paraphrases
import pandas as pd
from tqdm import tqdm
import argparse
import json
from openai import OpenAI

file_path = 'raw_data/do_not_answer.csv'

df = pd.read_csv(file_path)

full_path = "Safety_Contrast/paraphrase_base.jsonl"

client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

with open(full_path, mode = "w", encoding="utf-8") as wf:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        category = row['types_of_harm']
        question = row['question']
        id = row['id']
        paraphrased_element = get_5_paraphrases(question, client, 1, category)
        result = {"id": id, "category": category, "question": question, "paraphrased_element": paraphrased_element}
        result = json.dumps(result)
        wf.write(str(result) + "\n")