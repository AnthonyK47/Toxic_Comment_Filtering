import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

import W2V_LSTM_config as config
from W2V_LSTM_preprocessing import clean_text, Vocabulary, train_word2vec, create_embedding_matrix
from W2V_LSTM_model import Word2VecLSTM
from W2V_LSTM_dataset import ToxicCommentDataset


def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        
        optimizer.zero_grad()                               # Zero gradients
        
       
        outputs = model(sequences)                          # Forward pass
        loss = criterion(outputs, labels.unsqueeze(1))
        
        
        loss.backward()                                     # Backward pass
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(sequences)
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds).flatten(), np.array(all_labels)


def main():
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    print(f"\nLoading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, usecols=["comment_text", "toxic"])
    
    print(f"Original size: {len(df)}")
    
    # Preprocessing
    df = df.dropna(subset=['comment_text', 'toxic'])                        # Remove missing values
    df = df[df['comment_text'].str.strip() != '']                           # Remove empty comments
    df = df[df['comment_text'].str.len() < 5000]                            # Remove extremely long comments
    df['comment_text'] = df['comment_text'].apply(clean_text)               # Apply text cleaning 
    df = df[df['comment_text'].str.strip() != '']                           # Remove any that became empty after cleaning
    
    print(f"After cleaning: {len(df)}")
    
    # Sample data if specified
    if config.SAMPLE_SIZE is not None and len(df) > config.SAMPLE_SIZE:
        df = df.sample(n=config.SAMPLE_SIZE, random_state=config.RANDOM_STATE)
        print(f"Using {len(df)} samples")
    
    # Train/validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['comment_text'].values,
        df['toxic'].values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df['toxic'].values
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    
    vocab = Vocabulary(max_vocab_size=config.MAX_VOCAB_SIZE)
    vocab.build_vocab(train_texts)                                                              # Build vocabulary
    
    
    word2vec_model = train_word2vec(train_texts, embedding_dim=config.EMBEDDING_DIM)            # Train Word2Vec
    
    
    embedding_matrix = create_embedding_matrix(word2vec_model, vocab, config.EMBEDDING_DIM)     # Create embedding matrix
    
    # Create datasets
    train_dataset = ToxicCommentDataset(train_texts, train_labels, vocab, config.MAX_LEN)
    val_dataset = ToxicCommentDataset(val_texts, val_labels, vocab, config.MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0  
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = Word2VecLSTM(
        vocab_size=len(vocab.word2idx),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        pretrained_embeddings=embedding_matrix
    )
    model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)   
    
    best_auc = 0

    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)     # Train
        print(f"Training loss: {train_loss:.4f}")
        
        
        val_preds, val_labels = evaluate(model, val_loader, device)                     # Evaluate
        
        
        val_preds_binary = (val_preds >= 0.5).astype(int)                               # Calculate metrics
        
        accuracy = accuracy_score(val_labels, val_preds_binary)
        precision = precision_score(val_labels, val_preds_binary)
        recall = recall_score(val_labels, val_preds_binary)
        f1 = f1_score(val_labels, val_preds_binary)
        auc = roc_auc_score(val_labels, val_preds)
        
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'word2vec_model': word2vec_model,
                'config': {
                    'embedding_dim': config.EMBEDDING_DIM,
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_layers': config.NUM_LAYERS,
                    'max_len': config.MAX_LEN,
                    'vocab_size': len(vocab.word2idx)
                },
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc
                }
            }, config.MODEL_PATH)
            print(f"Best model (AUC: {auc:.4f})")
    
    print(f"Training complete! Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {config.MODEL_PATH}")
    
if __name__ == "__main__":
    main()