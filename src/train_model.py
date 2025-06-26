from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from data_preprocessing import load_and_prepare_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tok_data, tokenizer = load_and_prepare_dataset(tokenizer_checkpoint=checkpoint)
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_data["train"],
        eval_dataset=tok_data["test"],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Save model and tokenizer
    trainer.save_model("saved_model")
    tokenizer.save_pretrained("saved_model")

if __name__ == "__main__":
    train()
