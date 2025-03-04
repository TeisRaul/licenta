import requests

file_url = "https://www.drpciv-romania.ro/Code/Applications/web/index.cgi?action=codulrutier"

response = requests.get(file_url)

if response.status_code == 200:
    lines = response.text.splitlines()
    for line in lines:
        print(line)
else:
    print("Failed to download file. Status code: ", response.status_code)