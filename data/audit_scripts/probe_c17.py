import urllib.request
import json

ep = "https://data.mendeley.com/api/datasets/2j8vvz3j6v/files"
req = urllib.request.Request(ep, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print("Total files in C17:", len(data))
    total_size = sum(item.get('size', 0) for item in data) / (1024*1024)
    print(f"Total size: {total_size:.2f} MB")
    extensions = {}
    for item in data:
        fn = item.get('filename', '')
        ext = fn.split('.')[-1].lower() if '.' in fn else 'none'
        extensions[ext] = extensions.get(ext, 0) + 1
    print("File extensions:", extensions)
