#!/usr/env/bin bash

# MODEL=YOUR_MODEL_HERE
# TOP_K=INSERT_TOP_K_INTEGER

MODEL=sentence-bert-paraphrase
TOP_K=10

# sentence-bert-paraphrase, sentence-bert-all, st5-xxl, st5-xl, st5-large, st5-base, llm2vec-mistral, llm2vec-llama3, sup-simcse, unsup-simcse
# top_k default=10

python -m Safety_Taxonomy.get_embeddings $MODEL
python -m Safety_Taxonomy.get_categorical_purity $MODEL $TOP_K
python -m Safety_Taxonomy.get_tsne_visualization $MODEL $TOP_K