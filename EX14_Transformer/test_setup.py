import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
print("\n[OK] Transformers imported")

from datasets import load_dataset
print("[OK] Datasets imported")

import numpy as np
print("[OK] NumPy imported")

from tqdm.auto import tqdm
print("[OK] tqdm imported")

print("\n" + "="*60)
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
print(f"[OK] Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

print("\n" + "="*60)
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print("[OK] Tokenizer loaded")

print("\nLoading model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"[OK] Model loaded on {device}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# Test tokenization
print("\nTesting tokenization...")
sample = dataset['train'][0]['text'][:100]
tokens = tokenizer(sample, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
print(f"[OK] Tokenization works! Input shape: {tokens['input_ids'].shape}")

# Test forward pass
print("\nTesting forward pass...")
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=tokens['input_ids'].to(device),
        attention_mask=tokens['attention_mask'].to(device)
    )
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
print(f"[OK] Forward pass works! Prediction: {'POSITIVE' if pred == 1 else 'NEGATIVE'}")
print(f"Probabilities: P(Neg)={probs[0][0]:.4f}, P(Pos)={probs[0][1]:.4f}")

print("\n" + "="*60)
print("*** ALL TESTS PASSED - Ready for training! ***")
print("="*60)
