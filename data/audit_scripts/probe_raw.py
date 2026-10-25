import urllib.request

req = urllib.request.Request('https://data.mendeley.com/datasets/4cr7f2359x/1', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        print("URL:", resp.geturl())
        content = resp.read().decode('utf-8', errors='ignore')
        print("Content sample:", content[:1000])
except Exception as e:
    print("Err:", e)
