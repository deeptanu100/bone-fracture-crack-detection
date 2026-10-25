import urllib.request
import json

ep = "https://data.mendeley.com/api/datasets/4cr7f2359x/files"
req = urllib.request.Request(ep, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    for item in data:
        fn = item.get('filename')
        fid = item.get('id')
        size_gb = item.get('size', 0) / (1024*1024*1024)
        size_mb = item.get('size', 0) / (1024*1024)
        print(f"BoneFract File: {fn} | Size: {size_gb:.2f} GB ({size_mb:.1f} MB) | ID: {fid}")
