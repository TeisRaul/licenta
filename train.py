import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from functions import bag_of_words, tokenize, stem
#from model import NeuralNet

filepaths = [
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\cod_rutier.txt",
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\Noulcodpenal.txt",
    r"C:\Users\teisr\.vscode\chatbot-romanian-laws\sources\noul-cod-penal-precizari_01022013.txt"
]

all_content = []
for filepath in filepaths:
    with open(filepath, "r", encoding="utf-8") as file:
        all_content.append(file.read())
        
print(all_content)