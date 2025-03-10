import random
import torch
import torch.nn as nn
from functions import bag_of_words, tokenize, stem
from classmodel import NeuralNet
import os
import re

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
data = torch.load(FILE, weights_only=True)

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

# Istoric conversație
conversation_history = []

def find_relevant_content(content, tokenized_sentence, history):
    lines = content.split("\n")
    relevant_lines = []
    article_pattern = r"Art\. \d+"
    law_pattern = r"Legea nr\. \d+/\d+"
    ignore_words = ["spune", "mi", "care", "este", "in", "despre", "ce", "si"]
    key_words = [word.lower() for word in tokenized_sentence if word.lower() not in ignore_words]
    
    article_request = any("articolul" in token.lower() for token in tokenized_sentence)
    if article_request:
        requested_article = None
        for token in tokenized_sentence:
            if re.match(r"\d+", token):
                requested_article = f"Art. {token}"
                break
        if requested_article:
            for i, line in enumerate(lines):
                if requested_article in line:
                    context = lines[i:i+4]
                    return context

    history_context = " ".join([entry['input'] + " " + entry['response'] for entry in history]).lower()
    if "mai mult" in " ".join(tokenized_sentence).lower() and history:
        last_input = history[-1]['input'].lower()
        last_response = history[-1]['response'].lower()
        key_words.extend(tokenize(last_input))  


    for i, line in enumerate(lines):
        article_match = re.search(article_pattern, line)
        law_match = re.search(law_pattern, line)
        if article_match or law_match:
            line_context = " ".join(lines[i:i+4]).lower()
            if all(word in line_context for word in key_words):
                return lines[i:i+4]
            
    sentence_str = " ".join(key_words)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if sentence_str in line_lower or (history_context and any(word in line_lower for word in key_words)):
            context = lines[i:i+4]  
            return context
        elif all(word in line_lower for word in key_words):
            relevant_lines.append(line.strip())
    
    return relevant_lines if relevant_lines else []

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    selected_content = all_content[tag]
    relevant_lines = find_relevant_content(selected_content, tokenized_sentence, conversation_history)
    
    if relevant_lines:
        response = "\n".join(relevant_lines[:4])
        print(f"{bot_name}: {response}")
    elif prob.item() > 0.3:
        file_name = os.path.basename(filepaths[tag])
        response = f"Am găsit informații în {file_name}, dar nu am detalii specifice despre '{sentence}'."
        print(f"{bot_name}: {response}")
    else:
        for content, filepath in zip(all_content, filepaths):
            relevant_lines = find_relevant_content(content, tokenized_sentence, conversation_history)
            if relevant_lines:
                response = "\n".join(relevant_lines[:4])
                file_name = os.path.basename(filepath)
                print(f"{bot_name}: {response} (din {file_name})")
                break
        else:
            response = "Nu înțeleg..."
            print(f"{bot_name}: {response}")

    conversation_history.append({
        'input': sentence,
        'response': response
    })