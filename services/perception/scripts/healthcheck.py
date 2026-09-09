"""Container health probe: the API answers and nothing critical is missing."""
import json
import sys
import urllib.request

url = "http://127.0.0.1:%s/v1/health" % __import__("os").environ.get("FG_PORT", "8000")
try:
    with urllib.request.urlopen(url, timeout=8) as response:
        payload = json.load(response)
except Exception as exc:
    print("health недоступен: %s" % exc)
    sys.exit(1)

# "degraded" is still serving — a deployment without a GPU is legitimate for
# re-running analytics over finished jobs, so only a dead API fails the probe
print(payload.get("status"))
sys.exit(0)
