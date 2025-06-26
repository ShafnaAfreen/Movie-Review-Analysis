# config.py

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
SUBSET_SIZE = 5000
SEED = 42

TRAINING_ARGS = {
    "output_dir": "./results",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1"
}
