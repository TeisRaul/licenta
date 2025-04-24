import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.inputs = []
        self.targets = []
        for i in range(0, len(token_ids) - block_size):
            chunk = token_ids[i : i + block_size + 1]
            self.inputs.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.targets.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_dataset(text, block_size=8):
    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokens = tokenizer.encode(text).ids
    dataset = TextDataset(tokens, block_size)
    return dataset, tokenizer

def load_data(folder, hf_datasets=None):
    data = ""

    # Fișiere locale
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            print(f" Procesare fișier: {filepath}")
            with open(filepath, encoding="utf-8") as f:
                data += f.read() + "\n"

    # Dataseturi Hugging Face
    if hf_datasets:
        for dataset_name in hf_datasets:
            print(f" Încărcare Hugging Face: {dataset_name}")
            ds = load_dataset(dataset_name, split="train")
            for item in ds:
                # Folosește coloana "text", "content" sau orice alt câmp disponibil
                text = item.get("text") or item.get("content") or ""
                data += str(text) + "\n"

    return data
