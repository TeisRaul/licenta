from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def train_tokenizer_from_folder(folder, vocab_size=10700):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])

    files = [open(os.path.join(folder, f), encoding='utf-8').readlines() for f in os.listdir(folder) if f.endswith(".txt")]
    lines = [line.strip() for file in files for line in file]
    
    tokenizer.train_from_iterator(lines, trainer)
    tokenizer.save("tokenizer.json")
    return tokenizer
