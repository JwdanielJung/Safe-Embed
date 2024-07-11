import sys
import os
from utils import parse_paraphrase_data
import pandas as pd
from tqdm import tqdm
import argparse
import json
from openai import OpenAI


file_path = 'Safety_Contrast/paraphrase_base.jsonl'
full_path = "Safety_Contrast/paraphrase_parsed.jsonl"

df = pd.read_json(file_path, lines=True)

prompt_types = ['prompt_#1','prompt_#2','prompt_#3','prompt_#4','prompt_#5']

with open(full_path, mode = "w", encoding="utf-8") as wf:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        paraphrase_dict = parse_paraphrase_data(row)
        paraphrased_prompts = []

        for i in range(5):
            paraphrased_prompts.append(paraphrase_dict[prompt_types[i]]['modified_prompt'])

        result = { "id": row['id'], "category": row['category'], "original_prompt": row['question'], "prompt_#1": paraphrased_prompts[0] , "prompt_#2": paraphrased_prompts[1],"prompt_#3": paraphrased_prompts[2],"prompt_#4": paraphrased_prompts[3],"prompt_#5": paraphrased_prompts[4]}
        result = json.dumps(result)
        wf.write(str(result) + "\n")



