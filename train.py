import mmap
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
block_size = 8
batch_size = 4
max_iters = 10000
learning_rate = 3e-4
eval_iters = 250
n_embd = 384
n_layer = 4
n_head = 4
dropout = 0.2
datasets = [
    "Gargaz/Romanian_updated",
    "Gargaz/Romanian_better",
    "hcoxec/romanian_100k",
    "Gargaz/Romanian",
    "saillab/alpaca-romanian-cleaned"
]

# Încarcă toate dataset-urile
print("Încarcă dataseturile...")
loaded_datasets = [load_dataset(ds) for ds in datasets]

# Afișează dataset-urile încărcate
print("Dataseturi încărcate:")
for ds in datasets:
    print(f"- {ds}")

print(device)

folder_path = 'sources'
text = ""

def load_all_files():
    # Încarcă toate fișierele din folderul 'sources'
    text = ""
    loaded_files = set()  # Set pentru a preveni încărcarea fișierelor de 2 ori
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and filename not in loaded_files:  # Verifică dacă fișierul a fost deja încărcat
            file_path = os.path.join(folder_path, filename)
            print(f"- {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read() + "\n"
            loaded_files.add(filename)  # Adaugă fișierul în set pentru a preveni încărcarea ulterioară
    return text

text = load_all_files()

chars = sorted(set(text) | {"<unk>"})  # Adaugă simbolul <unk> în vocabular
vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int.get(c, string_to_int["<unk>"]) for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def load_all_files():
    # Încarcă toate fișierele din folderul 'sources'
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # doar fișierele .txt
            file_path = os.path.join(folder_path, filename)
            print(f"- {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read() + "\n"
    return text

text = load_all_files()

def get_random_chunk_from_dataset(dataset, split, block_size, batch_size):
    # Verifică split-urile disponibile
    print(f"Available splits: {dataset.keys()}")
    
    if split not in dataset.keys():
        print(f"Split {split} not found in dataset!")
        return None
    
    split_data = dataset[split]
    print(f"Dataset columns: {split_data.column_names}")
    
    # Alege corect coloana de text
    if 'input' in split_data.column_names:
        data = split_data['input']  # Folosește 'input' pentru text
    elif 'sentence' in split_data.column_names:
        data = split_data['sentence']
    elif 'text' in split_data.column_names:
        data = split_data['text']  # Dacă 'text' este disponibil
    else:
        print("Neither 'input', 'sentence', nor 'text' column found.")
        return None

    if data is None:
        print("Data is None!")
        return None

    print(f"Data shape: {len(data)}")  # Verifică lungimea datelor
    
    # Codifică textul în indici
    start_pos = random.randint(0, len(data) - block_size * batch_size)
    block = data[start_pos:start_pos + block_size * batch_size]
    
    # Asigură-te că datele sunt codificate corect
    encoded_data = encode(block)  # Codifică textul în indici
    return torch.tensor(encoded_data, dtype=torch.long)


def get_batch_from_dataset(dataset, split, block_size, batch_size):
    data = get_random_chunk_from_dataset(dataset, split, block_size, batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_random_chunk_from_sources_and_datasets(split, block_size, batch_size, source_type='txt'):
    if source_type == 'txt':
        # Extrage date din fișierele .txt
        start_pos = random.randint(0, len(text) - block_size * batch_size)
        block = text[start_pos:start_pos + block_size * batch_size]
        data = torch.tensor(encode(block), dtype=torch.long)
    elif source_type == 'dataset':
        # Extrage date dintr-un dataset
        dataset = random.choice(loaded_datasets)
        data = get_random_chunk_from_dataset(dataset, split, block_size, batch_size)
    
    return data

def get_batch_from_source(split, block_size, batch_size, source_type):
    available_splits = ['train', 'test']
    if split not in available_splits:
        print(f"Warning: {split} not found in dataset. Using 'train' split instead.")
        split = 'train'  # Fallback to 'train'

    # Get data from the specified split
    data = get_random_chunk_from_sources_and_datasets(split, block_size, batch_size, source_type)
    
    if data is None:
        print(f"Data is None for split: {split}")
        return None, None
    
    ix = torch.randint(len(data) - block_size, (batch_size,))  # Generate batch indices
    x = torch.stack([data[i:i + block_size] for i in ix])  # Get X batch
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])  # Get Y batch
    x, y = x.to(device), y.to(device)
    
    return x, y


def get_batch_from_sources_and_datasets(split, block_size, batch_size):
    data = get_random_chunk_from_sources_and_datasets(split, block_size, batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    print(f"x shape: {x.shape}, y shape: {y.shape}") 
    return x, y

def calculate_loss(logits, targets):
    # Asigură-te că logits sunt de dimensiuni (batch_size, seq_len, vocab_size)
    loss_fn = nn.CrossEntropyLoss()  # Folosește CrossEntropyLoss pentru modele de limbaj
    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))  # Flatten pentru a se potrivi cu dimensiunea de loss
    return loss

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_from_source(split, block_size, batch_size, source_type='dataset')
            if X is None or Y is None:
                print(f"Warning: Data is None for split: {split}")
                continue
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# [1, 0, 0]
# [1, 0.6, 0]
# [1, 0.6, 0.4]
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
# print('loading model parameters...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loaded successfully!')
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

sources = ['txt', 'dataset']  # Definim sursele (fișiere .txt și dataseturi)
# În bucla principală de antrenament
for iter in range(max_iters):
    print(f"Iterația {iter}")
    
    if iter % eval_iters == 0:
        # Estimează pierderea pentru train și val
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
    
    # Alege sursa aleatorie pentru date
    source_type = random.choice(sources)
    
    # Extrage un lot de date
    xb, yb = get_batch_from_source('train', block_size, batch_size, source_type)
    
    # Calculează pierderea pentru lotul curent
    logits, loss = model.forward(xb, yb)
    
    # Începe optimizarea
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Final Loss: {loss.item()}")

# Salvează modelul antrenat
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')
