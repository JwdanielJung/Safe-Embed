#!/usr/env/bin bash

MODEL=YOUR_MODEL_HERE
# sentence-bert-paraphrase, sentence-bert-all, st5-xxl, st5-xl, st5-large, st5-base, llm2vec-mistral, llm2vec-llama3, sup-simcse, unsup-simcse

python -m Safety_Contrast.get_embeddings $MODEL
python -m Safety_Contrast.get_boundary_similarity $MODEL