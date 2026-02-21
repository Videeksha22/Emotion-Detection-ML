import streamlit as st
import pickle
import re
import os

# Load model and vectorizer from model folder
model = pickle.load(open("model/emotion_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
encoder = pickle.load(open("model/emotion_encoder.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

st.set_page_config(page_title="Emotion Detection App")

st.title("🧠 Emotion Detection from Text")
st.write("Enter a sentence and the model will predict the emotion.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vector = tfidf.transform([cleaned_text])
        prediction = model.predict(vector)
        probability = model.predict_proba(vector)

        emotion_labels = {
            0: "joy",
            1: "sadness",
            2: "anger",
            3: "fear",
            4: "love",
            5: "surprise"
        }

        emotion = emotion_labels[prediction[0]]

        confidence = round(max(probability[0]) * 100, 2)

        emotion_emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😡",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲"
        }

        emoji = emotion_emojis.get(emotion, "")

        st.success(f"Predicted Emotion: {emotion} {emoji}")
        st.info(f"Confidence: {confidence}%")

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")

