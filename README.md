# Deep Bi-LSTM Text Classification Model for Fraud Detection

## Project Overview

This project involves building a **Deep Bidirectional LSTM (Bi-LSTM)** neural network using TensorFlow/Keras to classify whether a given textual data sample is **fraudulent or not**. It uses natural language processing (NLP) techniques to convert raw text into numerical sequences and trains a deep learning model to learn patterns indicative of fraudulent behavior.
Absolutely! Here's the updated `README.md` section with a dedicated explanation of **what a Bi-LSTM is**. You can insert this under a new section in the README:

---

### What is a Bi-LSTM?

A **Bidirectional Long Short-Term Memory (Bi-LSTM)** network is an extension of the traditional LSTM, which is a type of Recurrent Neural Network (RNN). Here's what makes it powerful:

#### LSTM (Long Short-Term Memory)

* LSTM is designed to **remember long-term dependencies** in sequence data.
* It uses **gates** (input, forget, and output) to control the flow of information and avoid problems like vanishing gradients that standard RNNs face.
* Ideal for text data where word order and context matter.

#### Bi-LSTM (Bidirectional LSTM)

* In a Bi-LSTM, the model has **two LSTM layers**:

  * One processes the sequence **forward** (left to right).
  * The other processes the sequence **backward** (right to left).
* This allows the model to learn **context from both directions**, capturing both **preceding** and **succeeding** words.

## Why Use a Deep Bi-LSTM for Fraud Detection?

Fraudulent cases often contain subtle cues and patterns in the text — such as unusual language, inconsistent phrasing, or specific keywords. These patterns are typically sequential and contextual, meaning that the **order and relationships between words** matter.

A **Bidirectional LSTM (Bi-LSTM)** is ideal for this task because:

- It can **understand context from both directions** (past and future).
- It handles **long-range dependencies** in sequences better than traditional models.
- It's robust for text classification tasks, especially where context and word ordering are crucial.

By using **multiple stacked Bi-LSTM layers**, we enhance the model’s ability to capture deep semantic patterns, making it suitable for distinguishing subtle differences between fraudulent and non-fraudulent texts.

---

## Dataset

The dataset used (`data.csv`) contains two columns:
- `TEXT`: The textual content of each transaction/report.
- `LABEL`: The class label indicating whether it is **fraudulent** or **not fraudulent**.

---

## Preprocessing Steps

1. **Label Encoding**: The text labels (`LABEL`) are converted into numerical format using `LabelEncoder`.
2. **Tokenization**: The `TEXT` data is tokenized into sequences of integers using TensorFlow's `Tokenizer`.
3. **Padding**: All sequences are padded to a uniform length (`maxlen=200`) to ensure consistent input dimensions for the model.

---

##  Model Architecture

The model is built using the **Sequential API** in TensorFlow Keras and consists of the following layers:

1. **Embedding Layer**  
   - Transforms word indices into dense vectors of size 128.
   - Captures semantic meaning of words.

2. **1st Bidirectional LSTM Layer**  
   - 128 LSTM units with dropout and recurrent dropout.
   - Returns full sequences to the next layer.

3. **2nd Bidirectional LSTM Layer**  
   - 64 LSTM units with dropout.
   - Extracts higher-level temporal features from the sequence.

4. **Dense Layer**  
   - 64 neurons with ReLU activation.
   - Learns complex patterns after sequence encoding.

5. **Dropout Layer**  
   - 40% dropout to prevent overfitting.

6. **Output Layer**  
   - Uses `softmax` activation to output class probabilities.
   - Number of units corresponds to number of classes (fraud vs. non-fraud).

---

## Model Training

- The model is trained for 10 epochs with a batch size of 32.
- `ModelCheckpoint` is used to save the best model based on training accuracy.
- Final model and best model are saved as `.h5` files.

---

## Saving Artifacts

To ensure the model can be used for predictions later:
- The **trained model** is saved (`deep_lstm_model_final.h5`).
- The **best-performing model** during training is checkpointed (`deep_lstm_model.h5`).
- The **tokenizer** and **label encoder** used during preprocessing are saved with `pickle` (`tokenizer.pkl` and `label_encoder.pkl`).

---

## How to Use

1. Place your `data.csv` file with `TEXT` and `LABEL` columns.
2. Run the script to preprocess the data, train the model, and save the results.
3. Use the saved model and tokenizer/label encoder to make predictions on new text data.

---

##  Use Case: Fraud Detection from Text

In many real-world scenarios (e.g., insurance claims, customer service reports, financial messages), fraud detection involves reading and interpreting text. This model automates the process by learning from previously labeled examples and making intelligent predictions about new, unseen reports.

By training on text data, this approach avoids the need for handcrafted features and leverages powerful deep learning to recognize subtle linguistic cues of fraud.

---

##  Requirements

- Python ≥ 3.7
- pandas
- numpy
- scikit-learn
- TensorFlow (2.x)
- pickle (for saving preprocessing tools)

---

##  Author

This project was developed as part of a fraud classification problem using natural language processing and deep learning.

---

##  Future Improvements

- Add validation split and track validation accuracy/loss.
- Use pretrained embeddings like GloVe or BERT for better text representation.
- Integrate explainability techniques (e.g., LIME or SHAP) to interpret predictions.

---
