import yaml
from sentence_transformers import SentenceTransformer, util
import torch
from openai import OpenAI
from dotenv import load_dotenv
import os
from transformers import AutoModel, AutoTokenizer
from llm2vec import LLM2Vec
from huggingface_hub import login
import warnings
from ast import literal_eval

warnings.filterwarnings("ignore")

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embedding(query, embedding_model):
    if embedding_model=="text-embedding-v3":
        client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        model="text-embedding-3-large"
        embedding = [x.embedding for x in client.embeddings.create(input = query, model=model).data]
    elif embedding_model == "sup-simcse":
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
        inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.tolist()
    elif embedding_model == "unsup-simcse":
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
        model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased").to(device)
        inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.tolist()
    elif embedding_model == "sentence-bert-paraphrase":
        model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2").to(device)
        embedding = model.encode(query).tolist()
    elif embedding_model == "sentence-bert-all":
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
        embedding = model.encode(query).tolist()
    elif embedding_model == "st5-xxl":
        model = SentenceTransformer('sentence-transformers/sentence-t5-xxl').to(device)
        embedding = model.encode(query).tolist()
    elif embedding_model == "st5-xl":
        model = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(device)
        embedding = model.encode(query).tolist()
    elif embedding_model == "st5-large":
        model = SentenceTransformer('sentence-transformers/sentence-t5-large').to(device)
        embedding = model.encode(query).tolist()
    elif embedding_model == "st5-base":
        model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
        embedding = model.encode(query).tolist()
    elif embedding_model == "llm2vec-mistral":
        login(token=os.getenv('HUGGINGFACE_TOKEN'))
        l2v = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,)
        embedding = l2v.encode(query).tolist()
    elif embedding_model == "llm2vec-llama3":
        login(token=os.getenv('HUGGINGFACE_TOKEN'))
        l2v = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        embedding = l2v.encode(query).tolist()

    return embedding

def get_similarity(original_embedding, modified_embedding):
    similarity = util.cos_sim(original_embedding, modified_embedding).numpy()[0][0]
    return similarity

def normalize_similarity(original_similarity, embedding_model):
    with open('similarities.yaml', 'r') as file:
        data = yaml.safe_load(file)

    for avg_similarity in data['average_similarities']:
        if avg_similarity['model'] == embedding_model:
            avg_cos = avg_similarity['similarity']
            break
    norm_cos = (original_similarity - avg_cos)/(1-avg_cos)

    return norm_cos

def get_5_contrasts(original_sentence, client, template_id, category):
    with open('Safety_Contrast/gpt_templates.yaml', 'r') as file:
        data = yaml.safe_load(file)

    for template in data['contrast_templates']:
        if template['id'] == template_id:
            content = template['content']
            break

    for policy in data['Do_not_answer_policies']:
        if policy['category'] == category:
            explanation = policy['explanation']
            break

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "user", "content": content.format(sentence = original_sentence, category=category, category_explanation = explanation)}
        ],
        temperature = 0
    ).choices[0].message.content
    
    return response


def parse_contrast_data(element):
    contrast_element = element['contrasts']
    start_index = contrast_element.find('{\n  \"prompt_#1')
    end_index = contrast_element.find('\"}\n}')+5
    contrast_dict = literal_eval(contrast_element[start_index:end_index])
    return contrast_dict