from datasets import load_dataset
import pandas as pd
from itertools import product
from utils import get_embedding, get_similarity
import statistics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('embedding_model', type=str, help='embedding_model')

args = parser.parse_args()

embedding_model = args.embedding_model


dataset = load_dataset("PKU-Alignment/BeaverTails")['30k_train']

df = dataset.to_pandas()
df = df.drop_duplicates(subset=["prompt"])

safe_df = df[df['is_safe']==True]
unsafe_df = df[df['is_safe']==False]


safe_sample = safe_df.sample(n=500, random_state=42)
unsafe_sample = unsafe_df.sample(n=500, random_state = 42)

all_sample = pd.concat([safe_sample, unsafe_sample])

head_sample = all_sample.sample(n=500, random_state=42)
tail_sample = all_sample.drop(head_sample.index)

head_prompt_list = head_sample['prompt'].to_list()
tail_prompt_list = tail_sample['prompt'].to_list()

head_embedding_list = get_embedding(head_prompt_list, embedding_model)
tail_embedding_list = get_embedding(tail_prompt_list, embedding_model)

embedding_combinations = list(product(head_embedding_list, tail_embedding_list))

similarities = []

for combination in embedding_combinations:
    similairty = get_similarity(combination[0], combination[1])
    similarities.append(similairty)

print(embedding_model)
print("mean:", round(statistics.fmean(similarities),3))
print("median:", round(statistics.median(similarities), 3))
print("std:", round(statistics.stdev(similarities), 3))