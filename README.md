# Movie Reviews Classification with Convolutional Neural Network (CNNs) and Word Embeddings

## Project Overview

This project applies Convolutional Neural Networks (CNNs) with pre-trained word embeddings (GloVe) to classify IMDB movie reviews as either positive or negative. Three different CNN models were tested to determine the most effective architecture.

## Dataset

- **IMDB Movie Review Dataset** with 50,000 labeled reviews.
- **Train set**: 25,000 reviews
- **Test set**: 25,000 reviews
- Only the top 10,000 most frequent words are used, with each review padded to 100 words.

## Models Implemented

Three CNN architectures were tested:

1. **Model 1**: Single Conv1D layer (128 filters, kernel size 5) with dropout.
2. **Model 2**: Two Conv1D layers (256 and 128 filters, kernel sizes 5 and 3) with dropout.
3. **Model 3 (Final Model)**: Optimized Conv1D architecture with 128 filters and improved hyperparameters.

## Results

| Model   | Training Accuracy | Validation Accuracy | Test Accuracy |
| ------- | ----------------- | ------------------- | ------------- |
| Model 1 | 73.52%            | 67.63%              | 69.98%        |
| Model 2 | 95.51%            | 70.21%              | 72.15%        |
| Model 3 | 93.52%            | 79.98%              | 83.04%        |

## Installation & Usage

### Prerequisites

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn

### Setup

1. Clone the repository:
   ```sh
   git clone [YOUR_REPO_LINK]
   cd [REPO_FOLDER]
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the GloVe embeddings (100d) and place them in the working directory:
   ```sh
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```
4. Run the Jupyter Notebook to preprocess the data, train models, and evaluate results:
   ```sh
   jupyter notebook NLP_midterm_project.ipynb
   ```

### Replication Steps
To reproduce the exact results from the `.ipynb` file:

1. **Load GloVe Embeddings**:
   - Ensure `glove.6B.100d.txt` is in the same directory.
   - Run the `load_glove_embeddings()` function to build the embedding matrix.

2. **Preprocess Data**:
   - Tokenize and pad the dataset to 100 words.
   - Ensure the IMDB dataset is properly loaded using `imdb.load_data(num_words=10000)`.

3. **Train Models**:
   - Model 1: 5 epochs, batch size 128, no dropout.
   - Model 2: 10 epochs, batch size 64, dropout 0.5.
   - Model 3 (Final): 30 epochs, batch size 32, learning rate 0.0005.

4. **Evaluate Performance**:
   - Run the evaluation cells to compute accuracy, loss, precision, recall, and F1-score.
   - Ensure TensorFlow GPU acceleration is enabled for faster training.

## Repository Structure

- `NLP_midterm_project.ipynb` - Contains the model training, evaluation, and results.
- `README.md` - This file with instructions.
- `requirements.txt` - List of dependencies.
- `glove.6B.100d.txt` - Pre-trained embeddings file.

## Authors

- Mukhil

## License

This project is for educational purposes only.

