import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

from datasets import load_dataset
import time

print("\nLoading IMDB dataset (this may take 1-2 minutes)...")
start = time.time()

try:
    dataset = load_dataset("imdb")
    elapsed = time.time() - start
    print(f"[SUCCESS] Dataset loaded in {elapsed:.1f} seconds!")
    print(f"  Train: {len(dataset['train']):,}")
    print(f"  Test:  {len(dataset['test']):,}")

    # Test sample
    print(f"\nSample:")
    print(dataset['train'][0]['text'][:200])

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
