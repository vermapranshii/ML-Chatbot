import streamlit as st
import pandas as pd
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================
# Download Required NLTK Data
# =========================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="ML Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =========================
# Load Model Files
# =========================

@st.cache_resource
def load_model():

    with open("chatbot_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# =========================
# Load Dataset
# =========================

@st.cache_data
def load_data():

    return pd.read_csv("chatbot.csv")


df = load_data()

# =========================
# Text Preprocessing
# =========================

stemmer = PorterStemmer()

stop_words = set(
    stopwords.words("english")
)


def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Simple tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================
# Chatbot Response Function
# =========================

def chatbot_response(user_text):

    try:

        cleaned = clean_text(user_text)

        # Convert text into TF-IDF vector
        vector = vectorizer.transform([cleaned])

        # Predict intent
        intent = model.predict(vector)[0]

        # Confidence score
        confidence = model.predict_proba(vector).max()

        # Low confidence handling
        if confidence < 0.40:
            return (
                "Sorry, I am not sure about that. "
                "Please ask something related to AI, Machine Learning, Python, or Data Science."
            )

        # Get response from dataset
        responses = df[
            df["intent"] == intent
        ]["response"]

        if len(responses) > 0:
            return responses.iloc[0]

        return "Sorry, I couldn't find an appropriate response."

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# Session State
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Main UI
# =========================

st.title("🤖 Machine Learning Chatbot")

st.write(
    "Ask questions about Python, AI, Machine Learning, Deep Learning, NLP, and Data Science."
)

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
user_input = st.chat_input(
    "Type your message here..."
)

if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    answer = chatbot_response(user_input)

    # Save bot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)

# =========================
# Sidebar
# =========================

with st.sidebar:

    st.header("📌 About Chatbot")

    st.write("""
This chatbot uses:

✅ NLP Text Processing

✅ Stopword Removal

✅ Stemming

✅ TF-IDF Vectorization

✅ Logistic Regression

✅ Streamlit
""")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()