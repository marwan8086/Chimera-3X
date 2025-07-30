import numpy as np
import torch
import git_info.git_wiki_pub.pub
import git_info.git_wiki_pub.wiki
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, Trainer, TrainingArguments
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model_classification = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model_classification.to(device)

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
    similarities = np.array([max([roberta_similarity(q, r) for r in R]) for q in Q])
    return np.mean(similarities)

def redundancy(R):
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

def cross_validate(R, Q, ground_truth, n_splits=2):
    if len(R) != len(ground_truth):
        raise ValueError(f"Length mismatch: R has {len(R)} items, but ground_truth has {len(ground_truth)} items.")
    
    kf = KFold(n_splits=n_splits)
    scores = []

    for train_index, test_index in kf.split(R):
        R_train, R_test = [R[i] for i in train_index], [R[i] for i in test_index]
        ground_truth_train, ground_truth_test = [ground_truth[i] for i in train_index], [ground_truth[i] for i in test_index]
        
        best_params, best_value = optimize_parameters(R_train, Q, ground_truth_train)
        score = objective_function(R_test, Q, ground_truth_test, **best_params)
        scores.append(score)
    
    return np.mean(scores)
#_____________________________________________ Analysis _____________________________________________________________________
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
    # ______________________________________ F1-Score ___________________________________________________________________________
    y_true = [1 if r in ground_truth else 0 for r in R]
    y_pred = [1 if roberta_similarity(q, r) > 0.5 else 0 for q in Q for r in R]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1-Score: {f1}")

def process_in_batches(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = get_embeddings(batch_texts)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def redundancy_batch(R, batch_size=16):
    embeddings = process_in_batches(R, batch_size)
    num_embeddings = len(embeddings)
    if num_embeddings < 2:
        return 0.0
    distances = [cosine(embeddings[i], embeddings[j]) for i in range(num_embeddings) for j in range(i + 1, num_embeddings)]
    return np.mean(distances)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'labels': torch.tensor(label, dtype=torch.long)}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy_score(labels, preds), 'precision': precision, 'recall': recall, 'f1': f1}

def train_model(train_texts, train_labels, tokenizer):
    dataset = CustomDataset(train_texts, train_labels, tokenizer)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model_classification,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

if __name__ == "__main__":
    R = [
        "The treatment for Alzheimer's includes medications and therapy.",
        "Cognitive therapies are essential."
    ]
    Q = ["What are the treatments for Alzheimer's disease?"]
    ground_truth = [
        "The treatment for Alzheimer's includes medications and therapy.",
        "Cognitive therapies are essential."
    ]
    labels = [1, 0]  

    if len(R) != len(ground_truth):
        raise ValueError(f"Length mismatch: R has {len(R)} items, but ground_truth has {len(ground_truth)} items.")

    objective_value = objective_function(R, Q, ground_truth)
    logging.info(f"Objective Value: {objective_value}")

    best_params, best_value = optimize_parameters(R, Q, ground_truth)
    logging.info(f"Best Parameters: {best_params}")
    logging.info(f"Best Objective Value: {best_value}")

    cv_score = cross_validate(R, Q, ground_truth, n_splits=2)
    logging.info(f"Cross-Validation Score: {cv_score}")

    analyze_results(R, Q, ground_truth)

    train_model(R, labels, tokenizer)
