import numpy as np
import torch
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki
from transformers import RobertaTokenizer, RobertaModel
from scipy.spatial.distance import cosine
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from SPARQLWrapper import SPARQLWrapper, JSON
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy() 
    return embeddings


def roberta_similarity(text1, text2):
    embeddings = get_embeddings([text1, text2])
    
    if embeddings.shape[0] != 2:
        raise ValueError("Expected 2 embeddings but got: {}".format(embeddings.shape[0]))
    
    similarity_score = 1 - cosine(embeddings[0], embeddings[1])
    return similarity_score

def accuracy(R, ground_truth):
    return np.mean([1 if r in ground_truth else 0 for r in R])

def relevance(Q, R):
    if not R:
        return 0
    similarities = np.array([max([roberta_similarity(q, r) for r in R]) for q in Q])
    return np.mean(similarities)

def redundancy(R):
    if not R:
        return 0
    embeddings = get_embeddings(R)
    
    num_embeddings = len(embeddings)
    if num_embeddings < 2:
        return 0.0
    
    distances = [cosine(embeddings[i], embeddings[j]) for i in range(num_embeddings) for j in range(i + 1, num_embeddings)]
    return np.mean(distances)

def objective_function(R, Q, ground_truth, alpha=0.5, beta=0.3, gamma=0.2):
    acc = accuracy(R, ground_truth)
    rel = relevance(Q, R)
    red = redundancy(R)
    return alpha * acc + beta * rel - gamma * red

def optimize_parameters(R, Q, ground_truth):
    param_grid = {
        'alpha': [0.3, 0.5, 0.7],
        'beta': [0.2, 0.3, 0.5],
        'gamma': [0.1, 0.2, 0.3]
    }

    best_value = float('-inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        current_value = objective_function(R, Q, ground_truth, **params)
        if current_value > best_value:
            best_value = current_value
            best_params = params

    return best_params, best_value

def cross_validate(R, Q, ground_truth):
    if len(R) != len(ground_truth):
        raise ValueError(f"Length mismatch: R has {len(R)} items, but ground_truth has {len(ground_truth)} items.")
    
    loo = LeaveOneOut()
    scores = []

    for train_index, test_index in loo.split(R):
        R_train, R_test = [R[i] for i in train_index], [R[i] for i in test_index]
        ground_truth_train, ground_truth_test = [ground_truth[i] for i in train_index], [ground_truth[i] for i in test_index]
        
        best_params, best_value = optimize_parameters(R_train, Q, ground_truth_train)
        score = objective_function(R_test, Q, ground_truth_test, **best_params)
        scores.append(score)
    
    return np.mean(scores)

def analyze_results(R, Q, ground_truth):
    logging.info("Results Analysis:")
    
    logging.info("\nSample Results:")
    for r in R:
        logging.info(f"Retrieved: {r}")
    
    logging.info("\nGround Truth:")
    for gt in ground_truth:
        logging.info(f"Ground Truth: {gt}")

    for q in Q:
        similarities = [roberta_similarity(q, r) for r in R]
        logging.info(f"\nSimilarities for Query '{q}':")
        for idx, sim in enumerate(similarities):
            logging.info(f"Similarity with Response '{R[idx]}': {sim}")

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

sparql_query = """
SELECT ?disease ?diseaseLabel ?treatment ?treatmentLabel WHERE {
  ?disease wdt:P31 wd:Q12136.
  ?disease wdt:P2176 ?treatment.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
LIMIT 100
"""

sparql.setQuery(sparql_query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

medical_data = []
for result in results["results"]["bindings"]:
    disease = result["diseaseLabel"]["value"]
    treatment = result["treatmentLabel"]["value"]
    medical_data.append((disease, treatment))

R = [treatment for _, treatment in medical_data if treatment is not None][:2]  
Q = ["What are the treatments for Alzheimer's disease?"]
ground_truth = [
    "The treatment for Alzheimer's includes medications and therapy.",
    "Cognitive therapies are essential."
]

if len(R) != len(ground_truth):
    raise ValueError(f"Length mismatch: R has {len(R)} items, but ground_truth has {len(ground_truth)} items.")

objective_value = objective_function(R, Q, ground_truth)
logging.info(f"Objective Value: {objective_value}")

best_params, best_value = optimize_parameters(R, Q, ground_truth)
logging.info(f"Best Parameters: {best_params}")
logging.info(f"Best Objective Value: {best_value}")

cv_score = cross_validate(R, Q, ground_truth)
logging.info(f"Cross-Validation Score: {cv_score}")

analyze_results(R, Q, ground_truth)
