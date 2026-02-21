# 🧠 Emotion Detection using NLP & Machine Learning

A Machine Learning web application that predicts human emotions from text using Natural Language Processing (NLP) techniques and a trained classification model.

---

## 📌 Project Overview

This project classifies text into the following emotions:

- 😊 Joy  
- 😢 Sadness  
- 😡 Anger  
- 😨 Fear  
- ❤️ Love  
- 😲 Surprise  

The model is trained using TF-IDF vectorization and a supervised machine learning classifier.  
The application is deployed using Streamlit for real-time predictions.

---

## 🚀 Features

- Text preprocessing and cleaning
- TF-IDF feature extraction
- Multi-class emotion classification
- Confidence score display
- Interactive Streamlit web interface
- Pre-trained model loading using Pickle

---

## 🛠️ Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Streamlit  
- Pickle  

---

## 📂 Project Structure

```
Emotion-Detection-ML/
│
├── app.py                    # Streamlit web application
├── Emotion_final.ipynb       # Model training notebook
├── text_ds.txt               # Dataset
├── requirements.txt          # Required libraries
│
├── model/                    # Saved trained models
│   ├── emotion_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── emotion_encoder.pkl
│
└── README.md
```

---

## ⚙️ How the Model Works

1. Input text is cleaned (lowercased + punctuation removed)

2. Text is converted to numerical features using TF-IDF

3. Trained ML model predicts emotion

4. Confidence score is calculated using probability

5. Final result is displayed with emoji

---

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Emotion-Detection-ML.git
cd Emotion-Detection-ML
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📌 Key Learnings

* Importance of text preprocessing in NLP
* Feature engineering using TF-IDF
* Multi-class classification handling
* Model serialization using Pickle

---

## 📈 Future Improvements

* Add deep learning model (LSTM / BERT)
* Add confusion matrix visualization
* Add probability bar chart visualization
* Deploy with Docker
* Add user history logging
