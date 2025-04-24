import gradio as gr
import torch
from model import MiniGPT
from tokenizers import Tokenizer
from generate import generate

tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

model = MiniGPT(vocab_size)
model.load_state_dict(torch.load("gpt_legislativ.pt"))

def chat(prompt):
    return generate(model, tokenizer, prompt, max_new_tokens=50)

gr.Interface(fn=chat, inputs="text", outputs="text", title="Chatbot legislativ în română").launch()
