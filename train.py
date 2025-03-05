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
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = torch.tensor(y_train, dtype=torch.long)

num_epochs = 1000
batch_size = 32
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
print(f"Final loss: {loss.item()}")
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. File saved to {FILE}")