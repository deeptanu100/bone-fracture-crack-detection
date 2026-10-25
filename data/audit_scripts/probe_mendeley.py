import urllib.request
import re
import json

def check_mendeley(dataset_id, version=1):
    url = f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files"
    print(f"Checking {dataset_id}...")
    try:
        # Check Mendeley web page
        page_url = f"https://data.mendeley.com/datasets/{dataset_id}/{version}"
        req = urllib.request.Request(page_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            title = re.findall(r'<title>(.*?)</title>', html)
            print(f"Page title: {title}")
            # Look for JSON payload in script tags
            scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
            for s in scripts:
                if 'files' in s and 'download_url' in s:
                    print("Found file json blob!")
                if 'download' in s and 'zip' in s:
                    matches = set(re.findall(r'https?://[^\s\"\'<>]+', s))
                    for m in matches:
                        if 'download' in m or 'zip' in m:
                            print(f"Candidate file url: {m}")
    except Exception as e:
        print(f"Error checking {dataset_id}: {e}")

check_mendeley("4cr7f2359x", 1)
check_mendeley("2j8vvz3j6v", 1)
