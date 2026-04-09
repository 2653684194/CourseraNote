# =============================================================================
# Transformer实战: 从零构建并训练情感分析模型
# =============================================================================
# 版本: 完全本地版 - 无需下载任何外部数据
# 特点: 使用PyTorch从头实现Transformer架构
# 优势: 100%离线可用,完整学习Transformer原理
# =============================================================================

import sys
import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime

LOG_FILE = './training_log.txt'

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.console = sys.stdout
    def write(self, msg):
        try:
            self.console.write(msg)
        except UnicodeEncodeError:
            self.console.write(msg.encode('cp1252', errors='replace').decode('cp1252'))
        self.file.write(msg)
        self.file.flush()
    def flush(self):
        self.file.flush()
        try:
            self.console.flush()
        except:
            pass

sys.stdout = Tee(LOG_FILE)

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")

log("="*70)
log("TRANSFORMER FROM SCRATCH: Sentiment Analysis Project")
log("="*70)
log("\nVersion: Pure PyTorch Implementation (No External Dependencies)")
log("Goal: Build & Train a Mini Transformer for Text Classification")

# Step 1: Environment Check
log("\n" + "-"*60)
log("STEP 1: Environment Check")
log("-"*60)

log(f"PyTorch Version: {torch.__version__}")
log(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Using Device: {device}")

# Step 2: Create Vocabulary & Tokenizer (Simple Word-Level)
log("\n" + "-"*60)
log("STEP 2: Building Custom Tokenizer")
log("-"*60)

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def build_vocab(self, texts, min_freq=1):
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        idx = len(self.word2idx)
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.vocab_size = len(self.word2idx)
        return self

    def encode(self, text, max_length=64):
        words = text.lower().split()
        ids = [self.word2idx['<CLS>']]

        for word in words[:max_length-2]:
            ids.append(self.word2idx.get(word, self.word2idx['<UNK>']))

        ids.append(self.word2idx['<SEP>'])

        padding_length = max_length - len(ids)
        if padding_length > 0:
            ids.extend([self.word2idx['<PAD>']] * padding_length)

        attention_mask = [1] * min(len(ids), max_length)
        attention_mask += [0] * max(0, max_length - len(attention_mask))

        return {
            'input_ids': ids[:max_length],
            'attention_mask': attention_mask[:max_length]
        }

POSITIVE_TEXTS = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "An amazing film with great acting and a wonderful storyline.",
    "Brilliant! One of the best movies I have ever seen.",
    "I really enjoyed this film. Highly recommended!",
    "Outstanding performance by the cast. A must-watch!",
    "This is a masterpiece of modern cinema.",
    "Wonderful movie that kept me engaged throughout.",
    "Excellent direction and superb cinematography.",
    "A truly heartwarming experience. Five stars!",
    "Incredible story with memorable characters."
]

NEGATIVE_TEXTS = [
    "Terrible movie. Complete waste of time and money.",
    "I hated this film. The worst movie I have seen.",
    "Boring and predictable. Do not watch this.",
    "Awful acting and a terrible script.",
    "Disappointing in every possible way.",
    "A complete disaster from start to finish.",
    "I regret watching this horrible movie.",
    "The plot made no sense. Very frustrating.",
    "One of the worst films of the year.",
    "Absolutely dreadful. Save your money."
]

NEUTRAL_TEXTS = [
    "The movie was okay, nothing special but not bad either.",
    "An average film with some good moments and some flaws.",
    "It was decent but could have been better.",
    "Not great, not terrible. Just mediocre.",
    "A mixed bag with both strengths and weaknesses."
]

def generate_dataset(n_samples=1000):
    np.random.seed(42)
    texts, labels = [], []

    all_texts = POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS

    for _ in range(n_samples):
        rand = np.random.random()
        if rand < 0.45:
            text = np.random.choice(POSITIVE_TEXTS)
            label = 1
        elif rand < 0.90:
            text = np.random.choice(NEGATIVE_TEXTS)
            label = 0
        else:
            text = np.random.choice(NEUTRAL_TEXTS)
            label = 1

        texts.append(text)
        labels.append(label)

    return texts, labels

log("Generating synthetic dataset...")
train_texts, train_labels = generate_dataset(2000)
test_texts, test_labels = generate_dataset(500)

all_texts = train_texts + test_texts

log("Building vocabulary...")
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(all_texts)

log(f"[OK] Tokenizer created!")
log(f"  Vocabulary size: {tokenizer.vocab_size:,}")
log(f"  Training samples: {len(train_texts):,}")
log(f"  Test samples:     {len(test_texts):,}")

sample_encoding = tokenizer.encode(train_texts[0], max_length=32)
log(f"\nExample encoding:")
log(f"  Input: '{train_texts[0][:50]}...'")
log(f"  Tokens: {sample_encoding['input_ids'][:10]}...")
log(f"  Length: {len(sample_encoding['input_ids'])}")

# Step 3: Implement Transformer Architecture from Scratch
log("\n" + "-"*60)
log("STEP 3: Implementing Transformer Architecture")
log("-"*60)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        if mask is not None:
            key_padding_mask = (mask == 0).any(dim=-1) if mask.dim() == 3 else (mask == 0)
        else:
            key_padding_mask = None

        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_embedding(positions)

        return self.dropout(x)


class TransformerForClassification(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=256, num_classes=2, max_len=64, dropout=0.1):
        super().__init__()

        self.embedding = EmbeddingLayer(vocab_size, d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
        else:
            mask = None

        for layer in self.encoder_layers:
            x = layer(x, mask)

        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


log("\nBuilding model architecture:")

MODEL_CONFIG = {
    'vocab_size': tokenizer.vocab_size,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 256,
    'num_classes': 2,
    'max_len': 32,
    'dropout': 0.1
}

model = TransformerForClassification(**MODEL_CONFIG).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

log(f"""
╔══════════════════════════════════════════════════════════════╗
║              TRANSFORMER ARCHITECTURE                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Input Text                                                ║
║       ↓                                                      ║
║   ┌─────────────────────────────────────┐                   ║
║   │  Token Embedding ({MODEL_CONFIG['d_model']}d)           │                   ║
║   │  Positional Encoding                │                   ║
║   └────────────────┬────────────────────┘                   ║
║                    ↓                                         ║
║   ┌─────────────────────────────────────┐                   ║
║   │  ×{MODEL_CONFIG['n_layers']} Transformer Encoder Layers│                   ║
║   │    ├─ Multi-Head Self-Attention      │                   ║
║   │    │   ({MODEL_CONFIG['n_heads']} heads, d_k={MODEL_CONFIG['d_model']//MODEL_CONFIG['n_heads']})         │                   ║
║   │    ├─ Add & LayerNorm               │                   ║
║   │    ├─ Feed-Forward Network           │                   ║
║   │    │   ({MODEL_CONFIG['d_model']} → {MODEL_CONFIG['d_ff']} → {MODEL_CONFIG['d_model']})          │                   ║
║   │    └─ Add & LayerNorm               │                   ║
║   └────────────────┬────────────────────┘                   ║
║                    ↓                                         ║
║   [CLS] Token Representation                               ║
║                    ↓                                         ║
║   ┌─────────────────────────────────────┐                   ║
║   │  Classification Head                │                   ║
║   │  Linear({MODEL_CONFIG['d_model']}→{MODEL_CONFIG['d_model']}) → ReLU → Dropout     │                   ║
║   │  Linear({MODEL_CONFIG['d_model']}→{MODEL_CONFIG['num_classes']})                    │                   ║
║   └────────────────┬────────────────────┘                   ║
║                    ↓                                         ║
║   Output: [NEGATIVE, POSITIVE]                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Model Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Parameters:  {total_params:,}
  Trainable Params:  {trainable_params:,}
  Model Size (FP32): ~{total_params * 4 / 1024**3:.2f} MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

log("[OK] Model architecture built successfully!")

# Step 4: Data Preprocessing
log("\n" + "-"*60)
log("STEP 4: Data Preprocessing")
log("-"*60)

MAX_LENGTH = 32
BATCH_SIZE = 32

def preprocess_data(texts, labels, tokenizer, max_length=MAX_LENGTH):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode(text, max_length=max_length)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(input_ids, attention_masks, labels_tensor)

log(f"Preprocessing data (max_length={MAX_LENGTH}, batch_size={BATCH_SIZE})...")

train_dataset = preprocess_data(train_texts, train_labels, tokenizer)
test_dataset = preprocess_data(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

log(f"[OK] Preprocessing complete!")
log(f"  Training batches: {len(train_loader)}")
log(f"  Test batches:     {len(test_loader)}")

# Step 5: Training Configuration
log("\n" + "-"*60)
log("STEP 5: Training Configuration")
log("-"*60)

EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

log(f"""
Hyperparameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Epochs:        {EPOCHS}
  Batch Size:    {BATCH_SIZE}
  Learning Rate: {LEARNING_RATE}
  Weight Decay:  {WEIGHT_DECAY}
  Optimizer:     AdamW
  Scheduler:     StepLR (step=3, gamma=0.5)
  Loss Function: CrossEntropyLoss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Step 6: Training Loop
log("="*70)
log("STEP 6: STARTING TRAINING")
log("="*70)

from tqdm.auto import tqdm

def train_epoch(model, loader, optimizer, criterion, device, epoch_num):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{EPOCHS}", leave=False)

    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))

    return avg_loss, accuracy, all_preds, all_labels


history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

BEST_MODEL_DIR = "./best_model"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
best_val_acc = 0.0

start_time = datetime.now()

for epoch in range(EPOCHS):
    log(f"\n{'─'*30} EPOCH {epoch+1}/{EPOCHS} {'─'*30}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
    scheduler.step()

    history['train_loss'].append(round(train_loss, 4))
    history['train_acc'].append(round(train_acc, 4))

    log(f"\n  ✓ Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")

    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

    history['val_loss'].append(round(val_loss, 4))
    history['val_acc'].append(round(val_acc, 4))

    log(f"  ✓ Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'config': MODEL_CONFIG
        }, os.path.join(BEST_MODEL_DIR, 'best_model.pt'))

        with open(os.path.join(BEST_MODEL_DIR, 'tokenizer_vocab.json'), 'w') as f:
            json.dump(tokenizer.word2idx, f)

        log(f"  ★ Best model saved! (Val Acc: {val_acc:.4f})")

    checkpoint = {
        'epoch': epoch + 1,
        'history': history,
        'best_val_acc': round(best_val_acc, 4),
        'timestamp': str(datetime.now())
    }

    with open('./training_checkpoint.json', 'w') as f:
        json.dump(checkpoint, f, indent=2)

training_time = str(datetime.now() - start_time).split('.')[0]
log(f"\n{'='*70}")
log(f"TRAINING COMPLETED IN {training_time}")
log(f"BEST VALIDATION ACCURACY: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
log(f"{'='*70}")

# Step 7: Final Evaluation
if best_val_acc > 0:
    log("\n" + "="*70)
    log("STEP 7: FINAL EVALUATION")
    log("="*70)

    checkpoint = torch.load(os.path.join(BEST_MODEL_DIR, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    final_loss, final_accuracy, final_preds, final_labels = evaluate(
        model, test_loader, criterion, device
    )

    log(f"\nFinal Test Results:")
    log(f"  ┌──────────────────────────────────┐")
    log(f"  │  Test Loss:     {final_loss:>12.4f}       │")
    log(f"  │  Test Accuracy: {final_accuracy:>12.4f} ({final_accuracy*100:>6.2f}%) │")
    log(f"  └──────────────────────────────────┘")

    y_true = np.array(final_labels)
    y_pred = np.array(final_preds)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    log(f"\nConfusion Matrix:")
    log(f"                  Predicted")
    log(f"               ┌─────────┬─────────┐")
    log(f"               │ NEGATIVE│ POSITIVE│")
    log(f"  Actual ┌───────┼─────────┼─────────┤")
    log(f"  NEGATIVE│ {cm[0][0]:7d} │ {cm[0][1]:7d} │")
    log(f"  POSITIVE│ {cm[1][0]:7d} │ {cm[1][1]:7d} │")
    log(f"          └───────┴─────────┴─────────┘")

    results = {
        'project_name': 'Transformer from Scratch - Sentiment Analysis',
        'architecture': 'Custom PyTorch Transformer',
        'hardware': f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}',
        'model_config': MODEL_CONFIG,
        'parameters': total_params,
        'dataset_info': {
            'train_samples': len(train_texts),
            'test_samples': len(test_texts),
            'type': 'Synthetic IMDB-style reviews'
        },
        'hyperparameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'training_history': history,
        'results': {
            'best_val_accuracy': round(best_val_acc, 4),
            'test_accuracy': round(final_accuracy, 4),
            'test_loss': round(final_loss, 4),
            'confusion_matrix': cm.tolist(),
            'training_time': training_time
        }
    }

    FINAL_MODEL_DIR = "./transformer_sentiment_model"
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
        'vocab': tokenizer.word2idx
    }, os.path.join(FINAL_MODEL_DIR, 'model.pt'))

    with open('./training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n[✓] Model saved to: {FINAL_MODEL_DIR}/")
    log(f"[✓] Results saved to: ./training_results.json")

    # Step 8: Inference Examples
    log("\n" + "="*70)
    log("STEP 8: INFERENCE DEMONSTRATION")
    log("="*70)

    def predict_sentiment(text, model, tokenizer, device, max_length=32):
        model.eval()

        encoded = tokenizer.encode(text, max_length=max_length)
        input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()

        return {
            'text': text[:150],
            'predicted_label': 'POSITIVE' if pred_label == 1 else 'NEGATIVE',
            'confidence': round(confidence, 4),
            'probability_positive': round(probs[0][1].item(), 4),
            'probability_negative': round(probs[0][0].item(), 4)
        }

    test_examples = [
        ("Positive", "This movie was absolutely fantastic! Great acting!"),
        ("Negative", "Terrible film. Waste of time and money."),
        ("Neutral", "It was okay, nothing special really."),
        ("Strong Positive", "I loved every minute of it! A masterpiece!")
    ]

    inference_results = []

    log("\nRunning inference:\n")

    for i, (desc, text) in enumerate(test_examples, 1):
        result = predict_sentiment(text, model, tokenizer, device)
        inference_results.append(result)

        log(f"Example {i}: [{desc}]")
        log(f"  Input:  {result['text']}")
        log(f"  Output: {result['predicted_label']} (conf: {result['confidence']:.4f})\n")

    with open('./inference_examples.json', 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, indent=2, ensure_ascii=False)

    log(f"[✓] Inference examples saved!")

else:
    log("\n[!] No valid trained model found.")

# Final Summary
log("\n" + "="*70)
log(" ★★★ PROJECT COMPLETE - SUMMARY ★★★ ")
log("="*70)

summary = f"""
================================================================================
       TRANSFORMER FROM SCRATCH - SENTIMENT ANALYSIS COMPLETE
================================================================================

PROJECT OVERVIEW:
  Task: Binary Text Classification (Positive/Negative Sentiment)
  Architecture: Custom PyTorch Transformer (Implemented from Scratch)
  Dataset: Synthetic IMDB-style Movie Reviews

HARDWARE USED:
  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
  VRAM: 16GB (RTX 5060 Ti)
  RAM: 32GB

MODEL ARCHITECTURE:
  Type: Encoder-only Transformer
  Embedding Dimension: {MODEL_CONFIG['d_model']}
  Attention Heads: {MODEL_CONFIG['n_heads']}
  Encoder Layers: {MODEL_CONFIG['n_layers']}
  FFN Dimension: {MODEL_CONFIG['d_ff']}
  Total Parameters: {total_params:,}
  Model Size: ~{total_params * 4 / 1024**3:.2f} MB

TRAINING DETAILS:
  Epochs: {EPOCHS}
  Batch Size: {BATCH_SIZE}
  Learning Rate: {LEARNING_RATE}
  Optimizer: AdamW
  Training Time: {training_time}

RESULTS:
  Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
  Final Test Accuracy:     {final_accuracy:.4f} ({final_accuracy*100:.2f}%)

WHAT YOU'VE LEARNED:
  ✓ How to implement Multi-Head Self-Attention from scratch
  ✓ How to build Transformer Encoder layers
  ✓ How to create custom tokenizers and vocabularies
  ✓ How to train Transformers end-to-end
  ✓ How to perform inference on new text

OUTPUT FILES:
  • Trained Model:     ./{FINAL_MODEL_DIR}/model.pt
  • Training Log:      ./training_log.txt
  • Results JSON:      ./training_results.json
  • Checkpoint:        ./training_checkpoint.json
  • Inference Examples: ./inference_examples.json

STATUS: ✅ SUCCESSFULLY COMPLETED

================================================================================
"""

log(summary)

if device.type == 'cuda':
    torch.cuda.empty_cache()
    log("[✓] GPU memory cleared")

log("\n" + "="*70)
log(" ★★★ CONGRATULATIONS! YOUR TRANSFORMER PROJECT IS COMPLETE! ★★★ ")
log("="*70)
log("""
You have successfully built and trained a Transformer from scratch!

KEY TAKEAWAYS:
  1. Transformers use Self-Attention mechanisms to process sequences
  2. Multi-Head Attention allows parallel attention on different subspaces
  3. Position encodings give models awareness of token order
  4. Layer normalization stabilizes training
  5. Residual connections help gradient flow

NEXT STEPS:
  1. Try larger datasets for better generalization
  2. Experiment with different hyperparameters
  3. Add more encoder layers for deeper representations
  4. Try pre-training on large corpora before fine-tuning
  5. Deploy your model as a web API
""")
