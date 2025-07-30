import requests
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki

#______________________________________ Fetching articles from PubMed use youer the PubMed API _________________________
def fetch_pubmed_articles(query, max_results=5):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'xml'
    }
    response = requests.get(base_url, params=params)
    return response.text
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dim) 
    index.add(embeddings)
    return index

def search_faiss(query_embedding, index, k=5):
    D, I = index.search(query_embedding, k)  
    return I 

def main():
    pubmed_articles = fetch_pubmed_articles('cancer treatment', 10)
    
 
    articles = pubmed_articles.split("<DocSum>")[1:]  
    embeddings = []
    
    for article in articles:
        text = article.split("<Item Name='Title'>")[1].split("</Item>")[0]  
        embedding = embed_text(text).numpy()  
        embeddings.append(embedding)
    
    embeddings = np.vstack(embeddings)  
    
    index = create_faiss_index(embeddings)
    
     #__________Test_____________
    query = "What are the latest cancer treatments?"
    query_embedding = embed_text(query).numpy()  
    
    top_k_indices = search_faiss(query_embedding, index, k=5)
    
    for idx in top_k_indices[0]:
        print("Article Title: ", articles[idx].split("<Item Name='Title'>")[1].split("</Item>")[0])  

if __name__ == "__main__":
    main()
