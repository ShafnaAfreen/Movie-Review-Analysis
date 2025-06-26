from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def load_model_and_tokenizer(model_path="saved_model", tokenizer_checkpoint="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    labels = {0: "Negative", 1: "Positive"}
    return labels[pred_class], confidence

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    sample_text = "While initially meandering, the film ultimately coalesces into a poignant exploration of human resilience."
    prediction, confidence = predict_sentiment(sample_text, model, tokenizer)
    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
