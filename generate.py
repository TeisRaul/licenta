import torch
from model import MiniGPT
from tokenizers import Tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids])
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
        next_token = torch.argmax(logits[0, -1, :]).item()
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]])], dim=1)
    return tokenizer.decode(input_tensor[0].tolist())
