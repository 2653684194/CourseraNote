"""Quick check: PyTorch build and CUDA availability."""
import sys
import torch

print("Python:", sys.executable)
print("PyTorch version:", torch.__version__)
print("PyTorch file:", torch.__file__)

# Detect build type
v = torch.__version__.lower()
if "+cpu" in v:
    print("Build: CPU-only (+cpu in version)")
    print("")
    print("To use GPU (RTX 5060 Ti / CUDA 13.0), run in PowerShell:")
    print("  pip uninstall torch torchvision -y")
    print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130')
elif "+cu" in v or "cu" in v.split("+")[-1][:2]:
    print("Build: CUDA")
else:
    print("Build: unknown (check PyTorch install)")

print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    try:
        torch.cuda.init()
    except Exception as e:
        print("CUDA init error:", type(e).__name__, e)
