import openai
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from rank_bm25 import BM25Okapi
from SPARQLWrapper import SPARQLWrapper, JSON
import networkx as nx
import logging
from scipy.spatial.distance import cosine
import FAISS 
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki


logging.basicConfig(level=logging.INFO)
openai.api_key = "API_YOUER" 
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaModel.from_pretrained('roberta-base')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_roberta.to(device)
def get_gpt4_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error with GPT-4 API: {e}")
        return None

def get_embeddings(texts):
    inputs = tokenizer_roberta(texts, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model_roberta(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

def roberta_similarity(text1, text2):
    embeddings = get_embeddings([text1, text2])
    
    if embeddings.shape[0] != 2:
        raise ValueError(f"Expected 2 embeddings but got: {embeddings.shape[0]}")
    
    similarity_score = 1 - cosine(embeddings[0], embeddings[1])
    return similarity_score

def get_medical_data():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery("""
    SELECT ?disease ?diseaseLabel ?treatment ?treatmentLabel WHERE {
      ?disease wdt:P31 wd:Q12136.
      ?disease wdt:P2176 ?treatment.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 1000
    """)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "MyMedicalApp/1.0 (contact@example.com)")
    
    try:
        results = sparql.query().convert()
    except Exception as e:
        logging.error(f"Failed to retrieve data: {e}")
        return []
    
    medical_data = []
    for result in results["results"]["bindings"]:
        disease = result["diseaseLabel"]["value"]
        treatment = result["treatmentLabel"]["value"]
        medical_data.append((disease, treatment))
    
    logging.info(f"Retrieved medical data: {medical_data}")
    return medical_data

def create_knowledge_graph(medical_data):
    G = nx.Graph()
    for disease, treatment in medical_data:
        G.add_node(disease, type='disease')
        G.add_node(treatment, type='treatment')
        G.add_edge(disease, treatment)
    return G

def retrieve_from_graph(G, query):
    treatments = []
    query_lower = query.lower()
    for node in G.nodes:
        if G.nodes[node]['type'] == 'disease' and query_lower in node.lower():
            treatments.extend([n for n in G.neighbors(node) if G.nodes[n]['type'] == 'treatment'])
    
    logging.info(f"Retrieved treatments for query '{query}': {treatments}")
    return treatments

def bm25_retrieve(docs, query):
    tokenized_docs = [doc.split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split(" ")
    best_docs = bm25.get_top_n(tokenized_query, docs, n=5)  
    return best_docs

def generate_gpt_response(query):
    gpt4_response = get_gpt4_response(query)
    return gpt4_response

if __name__ == "__main__":
    medical_data = get_medical_data()
    if not medical_data:
        logging.error("No medical data retrieved, exiting.")
        exit()

    knowledge_graph = create_knowledge_graph(medical_data)
    logging.info(f"Knowledge graph nodes: {knowledge_graph.nodes(data=True)}")
    logging.info(f"Knowledge graph edges: {knowledge_graph.edges()}")
 #___________________Test______________________
    query = "diabetes"
    retrieved_responses = retrieve_from_graph(knowledge_graph, query)

    if not retrieved_responses:
        logging.error("No responses retrieved from graph.")
        exit()

    docs = [f"Disease: {disease}, Treatment: {treatment}" for disease, treatment in medical_data]
    bm25_responses = bm25_retrieve(docs, query)
    logging.info(f"BM25 responses for query '{query}': {bm25_responses}")

    gpt4_response = generate_gpt_response(query)
    logging.info(f"GPT-4 response for query '{query}': {gpt4_response}")
