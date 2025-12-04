import torch
from W2V_LSTM_train import config

checkpoint = torch.load(config.MODEL_PATH, weights_only=False)

print("=" * 60)
print("FINAL RESULTS - Word2Vec + LSTM")
print("=" * 60)

metrics = checkpoint['metrics']

print(f"\nAccuracy:  {metrics['accuracy']*100:.2f}%")
print(f"Precision: {metrics['precision']*100:.2f}%")
print(f"Recall:    {metrics['recall']*100:.2f}%")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
print(f"AUC:       {metrics['auc']:.4f}")

print("\n" + "=" * 60)























