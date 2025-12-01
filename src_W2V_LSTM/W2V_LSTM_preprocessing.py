import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text):

   
    text = re.sub(r'<.*?>', '', text)              # Remove HTML tags
    
    
    text = re.sub(r'http\S+|www\S+', '', text)     # Remove URLs
    
    text = ' '.join(text.split())                  # Remove extra whitespace
    
    return text


def tokenize_text(text):

    tokens = word_tokenize(text.lower())  
    return tokens


class Vocabulary:

    def __init__(self, max_vocab_size=50000):

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        self.max_vocab_size = max_vocab_size
        
    def build_vocab(self, texts):
        
        # Count all words
        for text in tqdm(texts, desc='Counting words'):
            tokens = tokenize_text(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)               # Keep most common words
        sorted_words = sorted_words[:self.max_vocab_size - 2]                                           # Reserve space for PAD and UNK
        
        # Build word2idx and idx2word dictionaries
        for idx, (word, count) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def text_to_sequence(self, text, max_len):

        tokens = tokenize_text(text)
        sequence = [self.word2idx.get(token, 1) for token in tokens]  # 1 is UNK
        
        # Pad or truncate to max_len
        if len(sequence) < max_len:
            sequence = sequence + [0] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        
        return sequence


def train_word2vec(texts, embedding_dim=100):
    
    tokenized_texts = [tokenize_text(text) for text in tqdm(texts, desc='Tokenizing')]          # Tokenize all texts
    
    # Train Word2Vec
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=embedding_dim,
        window=5,                   # Context window size
        min_count=2,                # Ignore words that appear less than 2 times
        workers=4,                  # Number of parallel workers
        epochs=10                   # Training epochs for Word2Vec
    )
    
    print(f"Word2Vec Vocabulary size: {len(model.wv)}")
    return model


def create_embedding_matrix(word2vec_model, vocab, embedding_dim):
    
    # Create empty matrix with one row for each word in vocabulary
    vocab_size = len(vocab.word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    found = 0

    # Loop through each word in vocabulary
    for word, idx in vocab.word2idx.items():

        if word in word2vec_model.wv:                                           # If W2V knows this word, use it's trained vector
            embedding_matrix[idx] = word2vec_model.wv[word]
            found += 1
        else:                                                                   # If W2V doesn't know this word, give it a random vector
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)     
    
    print(f"Found {found}/{vocab_size} words in Word2Vec model")
    return embedding_matrix