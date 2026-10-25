import urllib.request
import json

for dataset_id, name in [("4cr7f2359x", "BoneFract"), ("2j8vvz3j6v", "Human Bone Fracture C17")]:
    ep = f"https://data.mendeley.com/api/datasets/{dataset_id}/files"
    try:
        req = urllib.request.Request(ep, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"=== {name} ({dataset_id}) ===")
            for item in data:
                fn = item.get('filename')
                fid = item.get('id')
                size = item.get('size', 0) / (1024*1024)
                download_url = item.get('download_url') or f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files/{fid}/file_downloaded"
                print(f"File: {fn} | Size: {size:.2f} MB | Download: {download_url}")
    except Exception as e:
        print(f"Error {dataset_id}: {e}")
