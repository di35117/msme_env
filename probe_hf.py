import os, json
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(".env"))
token = os.environ.get("HF_TOKEN", "")

model = "Qwen/Qwen2.5-0.5B-Instruct"
urls = [
    f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
    "https://api-inference.huggingface.co/v1/chat/completions",
]

for url in urls:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "say OK"}],
        "max_tokens": 8,
        "stream": False,
    }).encode()
    req = Request(url, data=payload, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
            reply = data["choices"][0]["message"]["content"]
            print(f"OK  {url}")
            print(f"    reply: {reply!r}")
    except HTTPError as e:
        body = e.read().decode()[:200]
        print(f"ERR {url}")
        print(f"    HTTP {e.code}: {body}")
    except Exception as e:
        print(f"ERR {url}: {e}")
