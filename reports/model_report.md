# Model Report

## Training Results Summary

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score | Precision | Recall |
|-------|---------------|-----------------|----------|----------|-----------|--------|
| 1     | 0.2894        | 0.2362          | 0.9000   | 0.9004   | 0.8863    | 0.9150 |
| 2     | 0.1442        | 0.3254          | 0.9120   | 0.9104   | 0.9160    | 0.9049 |
| 3     | 0.0692        | 0.3963          | 0.9110   | 0.9098   | 0.9108    | 0.9089 |

{
'eval_loss': 0.3254144787788391,
'eval_accuracy': 0.912, 
'eval_f1': 0.9103869653767821, 
'eval_precision': 0.9159836065573771,
'eval_recall': 0.9048582995951417,
'eval_runtime': 14.5919,
'eval_samples_per_second': 68.531,
'eval_steps_per_second': 4.317,
'epoch': 3.0
}

- The model achieved the best accuracy (91.2%) and F1 score (0.91) on epoch 2.
- Validation loss increased slightly after epoch 1, but accuracy and F1 stayed stable, indicating good generalization.
- Precision and recall are balanced, showing the model effectively distinguishes positive and negative sentiments.
