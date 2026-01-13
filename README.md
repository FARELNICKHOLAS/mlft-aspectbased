Below is the English version of your `README.md`, with structure and technical meaning preserved.

---

# Aspect-Based Sentiment Analysis (ABSA) – TIX ID App Reviews

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

This project is an **Aspect-Based Sentiment Analysis (ABSA)** system specifically designed to classify user reviews of the **TIX ID** mobile application from the Google Play Store. The system adopts a *Deep Learning* approach using a **BiLSTM** architecture combined with **FastText** word embeddings to provide granular insights into specific service aspects and their sentiment polarity.

## Key Features

* **Single Prediction**: Manually input a review text to obtain instant predictions.
* **Batch Processing**: Upload CSV/Excel files to analyze thousands of reviews simultaneously.
* **Error Analysis**: Evaluation feature to compare model predictions against ground truth labels, complete with a *Confusion Matrix*.
* **Indonesian NLP Pipeline**: Comprehensive preprocessing pipeline (Text Cleaning, Slang Normalization, Stopword Removal, and Sastrawi Stemming).

---

## Model Architecture

The model is built using the following components:

1. **FastText Embedding**: 300-dimensional embeddings to effectively handle non-standard words (*slang*) and typographical errors.
2. **Bidirectional LSTM (BiLSTM)**: Captures contextual information from both forward and backward directions.
3. **SMOTE**: Data balancing technique to address class imbalance in the original dataset.

The system classifies reviews into **6 Main Aspects**:

* Account Access
* Service Coverage
* Cinema Ticket Services
* Payment Methods
* Application Updates
* Promotions and Discounts

---

## How to Run (Local Deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/FARELNICKHOLAS/mlft-aspectbased.git
cd mlft-aspectbased
```

### 2. Create and Activate a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## Evaluation Results

Based on testing on the *test set*, the model achieved the following performance:

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| **Sentiment** | 86%      | 86%       | 86%    | 86%      |
| **Aspect**    | 88%      | 81%       | 82%    | 82%      |

---

## Directory Structure

```text
.
├── app.py                     # Main Streamlit application script
├── bilstm_sentiment.h5        # Sentiment classification model
├── bilstm_aspect.h5           # Aspect classification model
├── tokenizer_sentiment.pkl    # Tokenizer for sentiment model
├── tokenizer_aspect.pkl       # Tokenizer for aspect model
├── label_encoder_aspect.pkl   # Label encoder for aspect categories
├── merged_slang_dict.json     # Slang normalization dictionary
├── stopwords-id.txt           # Indonesian stopwords list
├── requirements.txt           # Required libraries
└── README.md
```

---

## Group Members (Group 2)

1. **Wayan Farel Nickholas Sadewa** (2208561051)
2. **David Brave Moarota Zebua** (2208561063)
3. **I Made Treshnanda Mas** (2208561089)

**Course Instructor:** Dr. Anak Agung Istri Ngurah Eka Karyawati, S.Si., M.Eng.
**Course:** Machine Learning for Text (MLFT) – Informatics, Universitas Udayana.

---


