import urllib.request
import json

endpoints = [
    "https://data.mendeley.com/api/datasets/4cr7f2359x/1",
    "https://data.mendeley.com/api/datasets/4cr7f2359x/files",
    "https://data.mendeley.com/api/v1/datasets/4cr7f2359x",
    "https://prod-dsrc-automl-augmnt-u-data-packager.mendeley.com/datasets/4cr7f2359x/1",
    "https://data.mendeley.com/datasets/4cr7f2359x/1/files",
    "https://api.mendeley.com/datasets/4cr7f2359x"
]

for ep in endpoints:
    try:
        req = urllib.request.Request(ep, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read().decode('utf-8')
            print(f"SUCCESS {ep}: {data[:200]}")
    except Exception as e:
        print(f"FAIL {ep}: {e}")
