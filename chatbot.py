import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functions import bag_of_words, tokenize, stem
from classmodel import NeuralNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

filepaths = [
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\cod_rutier.txt",
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\Noulcodpenal.txt",
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\noul-cod-penal-precizari_01022013.txt"
]

with open(filepaths[0], "r", encoding="utf-8") as file:
    cod_rutier = file.read()

with open(filepaths[1], "r", encoding="utf-8") as file:
    noul_cod_penal = file.read()
    
with open(filepaths[2], "r", encoding="utf-8") as file:
    noul_cod_penal_precizari = file.read()
    
all_content = [cod_rutier, noul_cod_penal, noul_cod_penal_precizari]

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "LawyerAI"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for cod in all_content:
            if tag == all_content.index(cod):
                print(f"{bot_name}: {cod}")
    else:
        print(f"{bot_name}: I do not understand...")