# Toxic Comment Classification - BERT vs Word2Vec + LSTM

This project compares two approaches for detecting toxic comments: BERT (a pre-trained transformer) and Word2Vec + LSTM (a custom-built model).

## Project Structure
```
3832_final_project/
|── Project_presentation/
|   |── Slide Show.pdf                             # Copy of the slide deck
|   |── Toxic Comment Filtering Final Report.pdf   # Summary of the entire project
|── src_BERT/
│   |── Kaggle_BERT_Notebook.ipynb                 # BERT training code (Kaggle notebook), see notebook for what each block of code does
|── src_LSTM/
│   |── W2V_LSTM_config.py                         # Configuration settings
│   |── W2V_LSTM_model.py                          # LSTM architecture
│   |── W2V_LSTM_dataset.py                        # Dataset class
│   |── W2V_LSTM_preprocessing.py                  # Text cleaning and Word2Vec training
│   |── W2V_LSTM_train.py                          # Main training script
│   |── W2V_LSTM_results.py                        # View saved results
|                   
|── README.md
|── Requirements.txt
```

## Dependencies

Install these packages:
```bash
pip install pandas numpy torch scikit-learn transformers gensim nltk tqdm
```

For NLTK, also run once:
```python
import nltk
nltk.download('punkt')
```

## Dataset

Download the **Jigsaw Toxic Comment Classification** dataset from Kaggle:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Create a dataset folder and put `train.csv` in the `dataset/` folder (or update the path in the code).

## How to Run

### BERT Model (Run on Kaggle)

1. Upload `Kaggle_BERT_Notebook.ipynb` to Kaggle
2. Add the Jigsaw dataset to your notebook
3. Set accelerator to GPU
4. Run all cells or hit Save Version and set Version Type to Save & Run All, then in Advanced Settings set to GPU
5. Training will take about an hour
6. Results will produce at the bottom
7. Download `bert_model.bin` from the output folder

### Word2Vec + LSTM Model (Run Locally)

1. Make sure `train.csv` is in the `dataset/` folder
2. Run the training script:
```bash
   python src_W2V_LSTM/W2V_LSTM_train.py
```
3. Training will vary based on if you use GPU vs CPU. GPU will be faster though.
4. Model saves to `word2vec_lstm_model.pt`
5. To view results run `python src_W2V_LSTM/W2V_LSTM_results.py`

