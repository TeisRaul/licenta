import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functions import bag_of_words, tokenize, stem
from classmodel import NeuralNet

filepaths = [
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\cod_rutier.txt",
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\Noulcodpenal.txt",
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\noul-cod-penal-precizari_01022013.txt"
]

all_content = []
for filepath in filepaths:
    with open(filepath, "r", encoding="utf-8") as file:
        all_content.append(file.read())

all_words = []
tags = []
xy = []

for content in all_content:
    tag = all_content.index(content)
    tags.append(tag)
    for line in content.split("\n"):
        tokenized = tokenize(line)
        all_words.extend(tokenized)
        xy.append((tokenized, tag))
        
ignore_words = ["?", "!", ".", ","]
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "data points")
print(len(tags), "tags", tags)
print(len(all_words), "unique stemmed words", all_words)

X_train = []
y_train = []
for (sentence, tag) in xy:
    bag = bag_of_words(sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(tag)
    
X_train = np.array(X_train)
y_train = np.array(y_train)