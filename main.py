import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

st.title("Sentiment Analysis App")

# Load model and tokenizer (you can cache this)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

tokenizer, model = load_model()

# User input
text = st.text_input("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if text:
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get prediction
        predictions = F.softmax(outputs.logits, dim=-1)
        sentiment = "Positive" if predictions[0][1] > predictions[0][0] else "Negative"
        confidence = max(predictions[0]).item()
        
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2%}")
    else:
        st.write("Please enter some text to analyze.")
