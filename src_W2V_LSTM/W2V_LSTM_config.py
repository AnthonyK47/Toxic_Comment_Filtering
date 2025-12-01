EMBEDDING_DIM = 100      # Word2Vec embedding dimension
HIDDEN_DIM = 128         # LSTM hidden dimension
NUM_LAYERS = 2           # Number of LSTM layers
MAX_LEN = 128            # Maximum sequence length

TRAIN_BATCH_SIZE = 32    # Training batch size
VALID_BATCH_SIZE = 32    # Validation batch size
EPOCHS = 5               # Number of training epochs
LEARNING_RATE = 0.001    # Learning rate

MAX_VOCAB_SIZE = 50000   # Maximum vocabulary size

DATA_PATH = "dataset/train.csv"  # Path to training data
MODEL_PATH = "word2vec_lstm_model.pt"  # Where to save the model

SAMPLE_SIZE = 40000      # Number of samples to use (set to None for all data)
TEST_SIZE = 0.1          # Validation split ratio
RANDOM_STATE = 42        # Random seed for reproducibility