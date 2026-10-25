import urllib.request

url = "https://data.mendeley.com/public-files/datasets/4cr7f2359x/files/76c032f0-b8be-49b8-a943-d30d77ac62e4/file_downloaded"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        print("Content-Length:", resp.headers.get('Content-Length'))
        print("Content-Type:", resp.headers.get('Content-Type'))
        sample = resp.read(1024)
        print("Read first 1024 bytes successfully! Magic bytes:", sample[:4])
except Exception as e:
    print("Download test failed:", e)
