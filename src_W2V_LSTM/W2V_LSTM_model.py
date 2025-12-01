import torch
import torch.nn as nn


class Word2VecLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pretrained_embeddings=None):

        super(Word2VecLSTM, self).__init__()
        
        # Embedding layer
        if pretrained_embeddings is not None:
           
            self.embedding = nn.Embedding.from_pretrained(           # Use pre-trained Word2Vec embeddings
                torch.FloatTensor(pretrained_embeddings),
                freeze=False                                         # Allow fine-tuning during training
            )
        else:
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Random initialization if no pre-trained embeddings
        
        # Bidirectional LSTM, meaning it can read text both forward and backwards: https://www.youtube.com/watch?v=jGst43P-TJA
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,         # Dropout between LSTM layers
            bidirectional=True                            # Process sequence in both directions
        )
        
        
        self.dropout = nn.Dropout(0.3)                    # Dropout for regularization
        
        
        self.fc = nn.Linear(hidden_dim * 2, 1)            # Final classification layer
    
    def forward(self, x):
        
        embedded = self.embedding(x)                      # Get word embeddings

        lstm_out, (hidden, cell) = self.lstm(embedded)    # Pass through LSTM
        
        hidden_forward = hidden[-2, :, :]                 # Last layer forward direction
        hidden_backward = hidden[-1, :, :]                # Last layer backward direction
        
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1) # Concatenate forward and backward hidden states
        
        hidden_concat = self.dropout(hidden_concat)       # Apply dropout             
        
        output = self.fc(hidden_concat)                   # Final classification
        
        return output
