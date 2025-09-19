
import os, pandas as pd, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch

def load_dataset(path='data/resumes.csv'):
    return pd.read_csv(path)

def compute_embeddings(texts, tokenizer, model, device='cpu', batch_size=16):
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
            # mean pooling on token embeddings using attention mask
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            pooled = summed / counts
            pooled = pooled.cpu().numpy()
            embeddings.append(pooled)
    embeddings = np.vstack(embeddings)
    return embeddings

def main():
    os.makedirs('model_files', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    # Load curated dataset
    df = load_dataset('data/resumes.csv')
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    print('Loading tokenizer and model (DistilBERT)...')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')

    print('Computing embeddings...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = compute_embeddings(texts, tokenizer, model, device=device, batch_size=32)

    print('Training classifier...')
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Save pipeline objects: classifier + tokenizer + model config indicator
    joblib.dump({'clf': clf}, 'model_files/bert_clf.joblib')
    print('Saved classifier to model_files/bert_clf.joblib')

if __name__ == '__main__':
    main()
