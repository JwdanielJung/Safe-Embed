from purity import *
from visualization import *
import os
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--is_preprocessed', action=argparse.BooleanOptionalAction, default=False, help="Boolean flag to indicate if the data is preprocessed")

    args = parser.parse_args()

    print(f"is_preprocessed: {args.is_preprocessed}")

    raw = load_data('../raw_data/do_not_answer.csv')
    query = raw['question'].tolist()
    risk_category = raw['types_of_harm'].tolist()

    embedding_models = ['sentence-bert-paraphrase','sentence-bert-all','sup-simcse','unsup-simcse','st5-base', 'st5-large','st5-xl', 'st5-xxl', 'text-embedding-v3', 'llm2vec-mistral','llm2vec-llama3']

    if args.is_preprocessed == False:
        encoding(query, risk_category, embedding_models)
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
        # make t-sne
        df = pd.read_json('embeddings.jsonl',lines=True)
        path = 'tsne'
        if not os.path.exists(path):
            os.makedirs(path)

        plot_tsne_subplots(df, embedding_models) # 2x1 & 3x3 plots
        
        # for each model plot
        # for embedding_model in embedding_models:
        #     tsne_single_visualization(df, embedding_model)
        cp_df = get_category_purity(df, embedding_models)

    else:
        cp_df = pd.read_json('cp.jsonl', lines=True)

    path = 'chart'
    if not os.path.exists(path):
        os.makedirs(path)

    cp_bar_chart_visualization(cp_df)
    cp_heatmap_visualization(cp_df)
