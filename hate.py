
# ===================
# Imports
# ===================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

# ===================
# Load Data & Create Vocab
# ===================
train_df = pd.read_csv('/home/dse316/grp21/nlp/hate_train.csv').dropna()
val_df = pd.read_csv('/home/dse316/grp21/nlp/hate_val.csv').dropna()
train_texts = train_df['Sentence'].tolist()
train_labels = train_df['Tag'].astype(int).tolist()
val_texts = val_df['Sentence'].tolist()
val_labels = val_df['Tag'].astype(int).tolist()

def basic_tokenizer(x): return x.lower().split()
all_text = train_texts + val_texts
vocab = ['<PAD>', '<UNK>'] + [w for w, _ in Counter(' '.join(all_text).split()).most_common(10000)]
word2idx = {w: i for i, w in enumerate(vocab)}
def encode_text(text, max_len=64):
    tokens = basic_tokenizer(text)
    idxs = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    return idxs[:max_len] + [word2idx['<PAD>']] * (max_len - len(idxs))

# ===================
# DataLoader Classes
# ===================
class BasicTextDataset(Dataset):
    def __init__(self, texts, labels, max_len=64):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(encode_text(self.texts[idx], self.max_len)), torch.tensor(self.labels[idx])

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class BERTDataset(Dataset):
    def __init__(self, texts, labels, max_len=64):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        item = bert_tokenizer(
            self.texts[idx], add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': item['input_ids'].squeeze(0),
            'attention_mask': item['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

BATCH_SIZE = 32
train_basic_dl = DataLoader(BasicTextDataset(train_texts, train_labels), batch_size=BATCH_SIZE, shuffle=True)
val_basic_dl = DataLoader(BasicTextDataset(val_texts, val_labels), batch_size=BATCH_SIZE)
train_bert_dl = DataLoader(BERTDataset(train_texts, train_labels), batch_size=16, shuffle=True)
val_bert_dl = DataLoader(BERTDataset(val_texts, val_labels), batch_size=16)

VOCAB_SIZE = len(word2idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Model Definitions
# =========================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_class=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb)
        return self.fc(hidden[-1]), hidden[-1]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_class=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
    def forward(self, x):
        emb = self.embedding(x)
        out, (hidden, _) = self.lstm(emb)
        return self.fc(hidden[-1]), hidden[-1]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=4, num_layers=2, num_class=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_class)
    def forward(self, x):
        emb = self.embedding(x).permute(1, 0, 2)
        out = self.transformer(emb).mean(dim=0)
        return self.fc(out), out

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
bert_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)

# =========================
# Training Loops
# =========================
def train_basic_model(model, train_dl, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for x, y in tqdm(train_dl, desc=f"Training {model.__class__.__name__}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model

def train_bert_model(model, train_dl, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_dl, desc="Training BERT"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

# =========================
# Train Models
# =========================
rnn_model = train_basic_model(RNNClassifier(VOCAB_SIZE), train_basic_dl)
lstm_model = train_basic_model(LSTMClassifier(VOCAB_SIZE), train_basic_dl)
trans_model = train_basic_model(TransformerClassifier(VOCAB_SIZE), train_basic_dl)
bert_model = train_bert_model(bert_model, train_bert_dl)

# =========================
# Embedding Extraction
# =========================
def get_preds_embs_basic(model, dl):
    model.eval()
    preds, labels, embs = [], [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits, emb = model(x)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            embs.append(emb.cpu())
    return np.array(preds), np.array(labels), torch.cat(embs)

def get_preds_embs_bert(model, encoder, dl):
    model.eval(); encoder.eval()
    preds, labels, embs = [], [], []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            enc_out = encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = enc_out.last_hidden_state[:, 0, :].cpu()
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
            embs.append(cls_emb)
    return np.array(preds), np.array(labels), torch.cat(embs)

# Get validation predictions for reporting only
rnn_preds, rnn_labels, _ = get_preds_embs_basic(rnn_model, val_basic_dl)
lstm_preds, lstm_labels, _ = get_preds_embs_basic(lstm_model, val_basic_dl)
trans_preds, trans_labels, _ = get_preds_embs_basic(trans_model, val_basic_dl)
bert_preds, bert_labels, _ = get_preds_embs_bert(bert_model, bert_encoder, val_bert_dl)

# Print validation classification results
print(f"RNN Validation F1: {f1_score(rnn_labels, rnn_preds):.4f}")
print(f"LSTM Validation F1: {f1_score(lstm_labels, lstm_preds):.4f}")
print(f"Transformer Validation F1: {f1_score(trans_labels, trans_preds):.4f}")
print(f"BERT Validation F1: {f1_score(bert_labels, bert_preds):.4f}")

# =========================
# Save Train Embeddings
# =========================
_, rnn_train_labels, rnn_train_embs = get_preds_embs_basic(rnn_model, train_basic_dl)
_, lstm_train_labels, lstm_train_embs = get_preds_embs_basic(lstm_model, train_basic_dl)
_, trans_train_labels, trans_train_embs = get_preds_embs_basic(trans_model, train_basic_dl)
_, bert_train_labels, bert_train_embs = get_preds_embs_bert(bert_model, bert_encoder, train_bert_dl)

torch.save({'embeddings': rnn_train_embs, 'labels': torch.tensor(rnn_train_labels)}, "rnn_train_embeddings.pt")
torch.save({'embeddings': lstm_train_embs, 'labels': torch.tensor(lstm_train_labels)}, "lstm_train_embeddings.pt")
torch.save({'embeddings': trans_train_embs, 'labels': torch.tensor(trans_train_labels)}, "trans_train_embeddings.pt")
torch.save({'embeddings': bert_train_embs, 'labels': torch.tensor(bert_train_labels)}, "bert_train_embeddings.pt")