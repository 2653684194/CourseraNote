import urllib.request
import ssl

print("Testing network connection to HuggingFace...")

try:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = "https://huggingface.co/datasets/imdb/resolve/main/dataset_info.json"
    print(f"Trying to access: {url}")

    response = urllib.request.urlopen(url, context=ctx, timeout=30)
    data = response.read().decode('utf-8')
    print(f"[SUCCESS] Connected to HuggingFace!")
    print(f"Response length: {len(data)} bytes")

except Exception as e:
    print(f"[ERROR] Network issue: {e}")
