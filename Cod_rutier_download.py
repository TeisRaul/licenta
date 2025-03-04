import requests

file_url = "https://www.drpciv-romania.ro/Code/Applications/web/index.cgi?action=codulrutier"

response = requests.get(file_url)

if response.status_code == 200:
    lines = response.text.splitlines()
    with open("cod_rutier.txt", "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line)
    print("File downloaded successfully")
else:
    print("Failed to download file. Status code: ", response.status_code)