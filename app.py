import streamlit as st
import pickle
import re
import nltk

nltk.download("stopwords") 
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load Saved Files
# -----------------------------

model = pickle.load(open("disease_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -----------------------------
# NLP Preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.  sub(r'[^a-zA-Z ]', ' ', text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# -----------------------------
# Prediction Function
# -----------------------------

def predict_disease(text):

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    return label_encoder.inverse_transform(prediction)[0]

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🏥 Clinical Trial Disease Category Classification")

st.markdown("""
This application predicts disease categories from clinical trial summaries using:

- NLP Preprocessing
- TF-IDF Vectorization
- Random Forest Classifier

**Model Accuracy:** 84.37%
""")
st.write(
    "Predict the disease category from a clinical trial summary using Machine Learning."
)

user_input = st.text_area(
    "Enter Clinical Trial Summary"
)

if st.button("Predict Disease"):

    if user_input.strip() != "":

        result = predict_disease(user_input)

        st.success(
            f"Predicted Disease Category: {result}"
        )

    else:

        st.warning(
            "Please enter a clinical trial summary."
        )