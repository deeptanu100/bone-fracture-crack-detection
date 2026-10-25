import urllib.request
import re
import json

for dataset_id, name in [("4cr7f2359x", "BoneFract"), ("2j8vvz3j6v", "Human Bone Fracture C17")]:
    req = urllib.request.Request(f'https://data.mendeley.com/datasets/{dataset_id}/1', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            desc = re.findall(r'<meta data-react-helmet="true" name="description" content="(.*?)"', content)
            print(f"=== {name} ({dataset_id}) ===")
            if desc:
                print("Description:", desc[0][:300], "...")
            # Extract download files/links or file size
            files_match = re.findall(r'"files":(\[.*?\])', content)
            if files_match:
                print("Files found in state JSON:")
                try:
                    files_data = json.loads(files_match[0])
                    for f in files_data:
                        print("File:", f.get("name"), "Size (MB):", round(f.get("size", 0)/(1024*1024), 2), "Download:", f.get("download_url"))
                except Exception as e:
                    print("Could not parse files json:", e)
    except Exception as e:
        print(f"Error for {dataset_id}: {e}")
