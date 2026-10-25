import urllib.request
import re

for dataset_id, name in [("4cr7f2359x", "BoneFract"), ("2j8vvz3j6v", "Human Bone Fracture C17")]:
    req = urllib.request.Request(f'https://data.mendeley.com/datasets/{dataset_id}/1', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
        print(f"=== {name} ({dataset_id}) ===")
        # Find any occurrence of .zip, .rar, size, etc.
        zips = set(re.findall(r'[\w\-/\.]+\.(?:zip|rar|tar\.gz|7z)', content, re.IGNORECASE))
        print("Archive matches:", zips)
        # Find size strings
        sizes = set(re.findall(r'\d+(?:\.\d+)?\s*(?:GB|MB|KB)', content))
        print("Size matches:", sizes)
        # Find any direct URLs
        urls = set(re.findall(r'https?://[^\s\"\'<>]+', content))
        dl_urls = [u for u in urls if 'download' in u or 'file' in u or 'mendeley' in u]
        print("Relevant URLs:", dl_urls[:5])
