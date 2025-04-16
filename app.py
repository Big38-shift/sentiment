import streamlit as st
import joblib
import string
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load model, vectorizer, and label encoder
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# Text preprocessing
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    tokens = [word.strip(string.punctuation) for word in text.split()]
    tokens = [word for word in tokens if word.isalpha()]
    stop = stopwords.words('english')
    tokens = [w for w in tokens if w not in stop]
    tokens = [w for w in tokens if len(w) > 1]
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(tokens)

# Streamlit UI
st.title("Hotel Review Sentiment Analysis")
st.write("Enter a hotel review and get its predicted sentiment!")

review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        sentiment = le.inverse_transform(prediction)[0]
        st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
# git