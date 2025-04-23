import requests
from bs4 import BeautifulSoup

file_url = "https://www.drpciv-romania.ro/Code/Applications/web/index.cgi?action=codulrutier"

response = requests.get(file_url)
response.encoding = "utf-8"

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    with open("cod_rutier.txt", "w", encoding="utf-8") as file:
        for line in text:
            file.write(line)
    print("File downloaded successfully")
else:
    print("Failed to download file. Status code: ", response.status_code)
    