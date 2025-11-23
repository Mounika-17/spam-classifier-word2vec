# 📧 Spam Message Classification using Word2Vec & Random Forest

This project is a **Spam Classifier** that identifies whether a given SMS/text message is **Spam** or **Ham (Not Spam)** using **Traditional Machine Learning + Word Embedding (Word2Vec)**.  
The model converts raw text into dense vector representations using **Average Word2Vec** and classifies using a **Random Forest Classifier**.

---

## 🚀 Project Overview

This project demonstrates how to build a spam detection model using **NLP techniques**, **feature engineering**, and **machine learning models**.

✔️ Text Cleaning & Preprocessing (stopwords removal, lemmatization, lowercasing)  
✔️ Feature Engineering using **Average Word2Vec Embeddings**  
✔️ Model Training using **Random Forest Classifier**  
✔️ Performance Evaluation using **Confusion Matrix, Accuracy, F1 Score**  
✔️ Lightweight, fast, and deployable model  

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|----------|--------------------|
| Programming Language | Python |
| Notebook | Jupyter Notebook |
| NLP & Text Processing | NLTK, Word2Vec, Gensim, Regex |
| Machine Learning | Scikit-Learn, RandomForestClassifier |
| Visualization | Matplotlib, Seaborn |
| Model Evaluation | Accuracy, F1 Score, Confusion Matrix |

---

## 📊 Algorithm Used

| Step | Description |
|------|-------------|
| Text Preprocessing | Clean text, remove punctuations, stopwords, and perform lemmatization |
| Feature Extraction | Average **Word2Vec embeddings** to convert text into numeric vectors |
| ML Algorithm | Random Forest Classifier for binary classification |
| Model Evaluation | Accuracy, F1 Score, Confusion Matrix |

---

## 📈 Model Performance

The model was evaluated using a test dataset of **1,114 messages**, including **Spam (1) and Ham (0)** categories.  

### Classification Report

              precision    recall  f1-score   support

           0       0.98      0.99      0.98       967
           1       0.91      0.86      0.88       147

    accuracy                           0.97      1114
   macro avg       0.95      0.92      0.93      1114
weighted avg       0.97      0.97      0.97      1114





### ✔ Random Forest for Spam Detection  

Random Forest performed well due to:  

Handling nonlinear patterns  

Reducing overfitting  

Fast training and easy deployment  


### 🔜 Future Improvements  

Deploy as a Flask/FastAPI Web App  

Fine-tune using BERT / DistilBERT (Transformers)  

Use LSTM / Bi-LSTM for sequence-based learning  

Hyperparameter tuning using GridSearchCV  



📌 Conclusion  

This project delivers a lightweight, interpretable, and efficient SMS Spam Detection model using Word2Vec and Random Forest, achieving high performance with minimal computational cost.

## 👩‍💻 Author  
Mounika Maradana  
📧 https://www.linkedin.com/in/mounikamaradana/  
🌐 https://github.com/Mounika-17  
