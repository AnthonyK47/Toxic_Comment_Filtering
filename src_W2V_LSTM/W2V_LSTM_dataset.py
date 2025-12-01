import torch
from torch.utils.data import Dataset


class ToxicCommentDataset(Dataset):

    def __init__(self, texts, labels, vocab, max_len):

        self.texts = texts                          # List or array of text strings
        self.labels = labels                        # List or array of labels (0 or 1)
        self.vocab = vocab                          # Vocabulary object with text_to_sequence method
        self.max_len = max_len                      # Maximum sequence length
    
    def __len__(self):
 
        return len(self.texts)                      # Return number of samples
    
    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        
        
        sequence = self.vocab.text_to_sequence(text, self.max_len) # Convert text to sequence of word indices
        

        # Return dictionary with sequence and label tensors
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }