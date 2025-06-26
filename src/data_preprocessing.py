from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_prepare_dataset(tokenizer_checkpoint="distilbert-base-uncased-finetuned-sst-2-english", subset_size=5000, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    dataset = load_dataset("imdb")
    df = dataset['train']
    shuffled_dataset = df.shuffle(seed=seed)
    subset_dataset = shuffled_dataset.select(range(subset_size))
    
    def tokenizefn(example):
        return tokenizer(example["text"], padding="longest", truncation=True)
    
    tok_data = subset_dataset.map(tokenizefn, batched=True)
    tok_data = tok_data.rename_column("label", "labels")
    tok_data.set_format("torch", columns=["input_ids", "labels", "attention_mask"])
    tok_data = tok_data.train_test_split(test_size=0.2, seed=seed)
    
    return tok_data, tokenizer
