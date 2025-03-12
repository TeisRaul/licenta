import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import re
from nltk.tokenize import word_tokenize
import nltk

# Descarcă resursele nltk (o singură dată)
nltk.download('punkt')

# Definește arhitectura modelului
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

# Încarcă datele din data.pth
data = torch.load('data.pth', map_location=torch.device('cpu'))

# Extrage parametrii
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

# Inițializează modelul
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data['model_state'])
model.eval()

# Funcții de procesare a textului
def tokenize(sentence):
    return word_tokenize(sentence.lower())

def bag_of_words(tokenized_sentence, all_words):
    bag = torch.zeros(len(all_words), dtype=torch.float32)
    for token in tokenized_sentence:
        if token in all_words:
            bag[all_words.index(token)] = 1.0
    return bag

# Inițializează serverul Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Primește mesajul utilizatorului
    user_message = request.json['message']
    
    # Procesează mesajul
    tokenized_message = tokenize(user_message)
    input_vector = bag_of_words(tokenized_message, all_words)
    
    # Rulează modelul
    with torch.no_grad():
        output = model(input_vector)
        _, predicted = torch.max(output, dim=0)
        predicted_tag = tags[predicted.item()]
    
    # Returnează rezultatul
    return jsonify({'predicted_tag': predicted_tag})

if __name__ == '__main__':
    app.run(host='192.168.0.115', port=5342)