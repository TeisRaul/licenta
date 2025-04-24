import time
import torch
import dataset
from model import MiniGPT
from dataset import load_data, create_dataset
from tokenizers import Tokenizer
from tokenizer import train_tokenizer_from_folder
from torch.utils.data import DataLoader

# Setări
FOLDER = "sources"
HF_DATASETS = [
    "Gargaz/Romanian_updated",
    "Gargaz/Romanian_better",
    "hcoxec/romanian_100k",
    "Gargaz/Romanian",
    "saillab/alpaca-romanian-cleaned"
]
BLOCK_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 3e-4

# 1. Creare tokenizer din fișiere locale
print(" Creare tokenizer...")
train_tokenizer_from_folder(FOLDER)

# 2. Încărcare date din fișiere și HuggingFace
print(" Încărcare date...")
text = load_data(FOLDER, hf_datasets=HF_DATASETS)

# 3. Creare dataset tokenizat
print(" Creare dataset...")
dataset, tokenizer = create_dataset(text, block_size=BLOCK_SIZE)

# Initialize DataLoader after BATCH_SIZE is defined
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Inițializare model și optimizator
vocab_size = tokenizer.get_vocab_size()
model = MiniGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Folosim dispozitivul: {device}")
model = MiniGPT(vocab_size).to(device)

# 5. Antrenare
print(" Start antrenare...")
for epoch in range(EPOCHS):
    start_time = time.time()
    print(time)
    total_loss = 0
    for i, batch in enumerate(loader):
        try:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            outputs = model(input_ids)
            print(f"[{i}] Epoca {epoch + 1}/{EPOCHS}: output: {outputs.shape}, target: {target_ids.shape}")

            loss = loss_fn(outputs.view(-1, vocab_size), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        except Exception as e:
            print(f"Eroare la batch {i}: {e}")
            raise
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ Epoca {epoch + 1} terminată. Loss total: {total_loss:.4f}. Timp: {elapsed:.2f} secunde.")

# 6. Salvare model
torch.save(model.state_dict(), "gpt_legislativ.pt")
print("Model salvat ca gpt_legislativ.pt")
print(" Antrenare finalizată!")
