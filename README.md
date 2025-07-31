# 🎬 IMDb Sentiment Classification

A PyTorch-based pipeline for **binary sentiment analysis** on the IMDb movie-review dataset. We compare three architectures:

- **SimpleClassifier** (GloVe embeddings frozen; feed-forward)
- **RNNClassifier** (GloVe embeddings fine-tuned; vanilla RNN)
- **LSTMClassifier** (GloVe embeddings fine-tuned; 2-layer LSTM)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Getting Started](#getting-started)  
3. [Dataset & Preprocessing](#dataset--preprocessing)  
4. [Embeddings](#embeddings)  
5. [Model Architectures](#model-architectures)  
6. [Training Details](#training-details)  
7. [Results](#results)  
8. [Conclusions](#conclusions)  
9. [Future Work](#future-work)  
10. [License](#license)  

---

## 📈 Project Overview

Train and evaluate three neural classifiers on 25 K training / 25 K testing IMDb reviews.  
**Goal**: distinguish _positive_ vs. _negative_ sentiment.

---

## 🚀 Getting Started

1. **Clone**  
   ```bash
   git clone https://github.com/yourname/imdb-text-classification.git
   cd imdb-text-classification


2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Download GloVe**

   * Get `glove.6B.100d.txt` from [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip)
   * Unzip and place in project root.
4. **Run**

   ```bash
   jupyter notebook TXTClasification.ipynb
   ```

---

## 🗂 Dataset & Preprocessing

* **Source**: HuggingFace `datasets.load_dataset("imdb")`.
* **Split**: 80 % train / 20 % validation (from the original 25 K train).
* **Tokenization**: NLTK’s `word_tokenize`.
* **Normalization**: lowercasing, removing punctuation & stopwords, lemmatization via POS tags.
* **Padding**: pad all sequences to max length 256 with `<pad>` index.

---

## 🔡 Embeddings

* **GloVe 100-dim**
* Build vocab only from tokens present in GloVe.
* Create an `embedding_matrix` of shape `(vocab_size, 100)`.

---

## 🏗 Model Architectures

| Model                | Embedding | RNN Type     | Layers | Hidden | Dropout |
| -------------------- | --------- | ------------ | ------ | ------ | ------- |
| **SimpleClassifier** | frozen    | –            | –      | 128    | 0.2     |
| **RNNClassifier**    | trainable | vanilla RNN  | 1      | 128    | 0.2     |
| **LSTMClassifier**   | trainable | 2-layer LSTM | 2      | 128    | 0.2     |

All feed into a 4-layer MLP with residual connections, batch-norm, and ReLU, ending in a single logit.

---

## 🏋️ Training Details

* **Loss**: `BCEWithLogitsLoss`
* **Optimizer**: `AdamW`
* **LR Schedule**: Cosine decay with 30 % warm-up
* **Batch size**: 32
* **Epochs**: 30 (Simple), 50 (RNN), 30 (LSTM)
* **Device**: GPU (if available)

---

## 📊 Results

### Validation (best epoch)

| Model                | Best Epoch | Val Loss | Val Acc | Val F1 |
| -------------------- | ---------- | -------- | ------- | ------ |
| **SimpleClassifier** | 9          | 0.4605   | 0.7836  | 0.7993 |
| **RNNClassifier**    | (≈35)      | –        | –       | ≈0.80  |
| **LSTMClassifier**   | (≈15)      | –        | –       | ≈0.84  |

<p align="center">
  <img src="figures/loss_acc_curves.png" alt="Loss & Accuracy curves" width="600"/>
</p>

### Final Test Performance

| Model                | Test Accuracy | Test F1    |
| -------------------- | ------------- | ---------- |
| **SimpleClassifier** | 0.7980        | 0.8009     |
| **RNNClassifier**    | 0.7983        | 0.8047     |
| **LSTMClassifier**   | **0.8311**    | **0.8496** |

> **Observation:** The 2-layer LSTM (fine-tuned embeddings) outperforms both the frozen feed-forward and vanilla RNN baselines by \~3 %–5 %.

---

## 🔍 Confusion Matrix (LSTM)

```python
from torchmetrics import BinaryConfusionMatrix
cf = BinaryConfusionMatrix().to(device)
# …after evaluating on test set…
print(cf.compute())
```

```
tensor([[9994, 1506],
        [1234,  926]] )
```

* True Negatives: 9 994
* False Positives: 1 506
* False Negatives: 1 234
* True Positives: 9 926

---

## 📝 Conclusions

* **Freezing embeddings** limits representational power (≈80 % F1).
* **Fine-tuning** with RNN gives marginal gain.
* **LSTM’s memory cells** capture sequence patterns ⇒ **best** at 0.8496 F1.
