import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load pre-trained model and vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("best_model.pkl", "rb") as model_file:
    best_model = pickle.load(model_file)

def clean_text(text):
    return text.lower().strip()

def predict_sentiment(message):
    message_cleaned = clean_text(message)
    message_vectorized = vectorizer.transform([message_cleaned])
    prediction = best_model.predict(message_vectorized)
    
    sentiment_label = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    return sentiment_label[prediction[0]]

def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")
    st.title("📊 Sentiment Analysis App")
    st.write("Enter text below to analyze its sentiment.")
    
    user_input = st.text_area("Enter text:", "")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input)
            
            st.markdown(f"### Sentiment: {sentiment}")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0,
                title={"text": "Sentiment Score"},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"},
                }
            ))
            st.plotly_chart(fig)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
