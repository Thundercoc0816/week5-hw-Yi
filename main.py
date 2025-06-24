import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
from collections import Counter
import pandas as pd
import numpy as np

st.title("Sentiment Analysis")
st.write("Analyze sentiment and see which words influence the prediction!")

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def preprocess_text(text):
    # Clean but preserve important punctuation for sentiment
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Remove # but keep the word
    text = ' '.join(text.split())
    return text.strip()

def get_word_importance(text, tokenizer, model):
    """Calculate importance of each word for sentiment prediction"""
    clean_text = preprocess_text(text)
    words = clean_text.split()
    
    if len(words) == 0:
        return [], []
    
    # Get baseline prediction for full text
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        baseline_outputs = model(**inputs)
    baseline_probs = F.softmax(baseline_outputs.logits, dim=-1)
    
    word_importances = []
    
    # Calculate importance by removing each word
    for i, word in enumerate(words):
        # Create text without this word
        text_without_word = ' '.join(words[:i] + words[i+1:])
        
        if text_without_word.strip():
            inputs_without = tokenizer(text_without_word, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs_without = model(**inputs_without)
            probs_without = F.softmax(outputs_without.logits, dim=-1)
            
            # Calculate difference in probabilities
            prob_diff = torch.abs(baseline_probs - probs_without).max().item()
            word_importances.append(prob_diff)
        else:
            word_importances.append(0)
    
    return words, word_importances

def classify_words_by_sentiment(words, importances, threshold=0.01):
    """Classify words as positive, negative, or neutral based on their importance"""
    positive_words = []
    negative_words = []
    neutral_words = []
    
    # Simple heuristic classification based on common sentiment words
    positive_indicators = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 
        'best', 'awesome', 'perfect', 'beautiful', 'happy', 'pleased', 'satisfied',
        'brilliant', 'outstanding', 'superb', 'magnificent', 'incredible', 'remarkable'
    }
    
    negative_indicators = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst', 'disgusting',
        'disappointing', 'sad', 'angry', 'frustrated', 'annoying', 'boring', 'stupid',
        'useless', 'pathetic', 'ridiculous', 'outrageous', 'unacceptable', 'dreadful'
    }
    
    for word, importance in zip(words, importances):
        word_lower = word.lower().strip('.,!?;:"')
        
        if importance > threshold:
            if word_lower in positive_indicators:
                positive_words.append((word, importance))
            elif word_lower in negative_indicators:
                negative_words.append((word, importance))
            else:
                neutral_words.append((word, importance))
        else:
            neutral_words.append((word, importance))
    
    return positive_words, negative_words, neutral_words

# Load model
tokenizer, model = load_model()

# User input
text = st.text_input("Enter text for sentiment analysis:", 
                    placeholder="e.g., I love this product but the delivery was terrible!")

if st.button(" Analyze Text") and text:
    # Basic word count statistics
    st.subheader(" Text Statistics")
    
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    unique_words = len(set(word.lower().strip('.,!?;:"') for word in words))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Words", word_count)
    with col2:
        st.metric("Characters", char_count)
    with col3:
        st.metric("Unique Words", unique_words)
    
    # Word frequency
    word_freq = Counter(word.lower().strip('.,!?;:"') for word in words)
    most_common = word_freq.most_common(5)
    
    st.write("**Most Frequent Words:**")
    freq_df = pd.DataFrame(most_common, columns=['Word', 'Count'])
    st.dataframe(freq_df, use_container_width=True)
    
    # Sentiment Analysis
    st.subheader(" Sentiment Analysis")
    
    clean_text = preprocess_text(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = F.softmax(outputs.logits, dim=-1)
    labels = ['Negative', 'Neutral', 'Positive']
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    # Display sentiment result
    sentiment_color = {"Negative": "", "Neutral": "", "Positive": ""}
    st.write(f"### {sentiment_color[labels[predicted_class]]} **Sentiment: {labels[predicted_class]}**")
    st.write(f"**Confidence: {confidence:.1%}**")
    
    # Show all probabilities
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Negative", f"{predictions[0][0].item():.1%}")
    with col2:
        st.metric("Neutral", f"{predictions[0][1].item():.1%}")
    with col3:
        st.metric("Positive", f"{predictions[0][2].item():.1%}")
    
    # Word Importance Analysis
    st.subheader(" Word Influence Analysis")
    st.write("See which words most influence the sentiment prediction:")
    
    with st.spinner("Analyzing word importance..."):
        words, importances = get_word_importance(text, tokenizer, model)
        
        if words and importances:
            # Create dataframe with word importance
            word_df = pd.DataFrame({
                'Word': words,
                'Importance': importances,
                'Influence': ['High' if imp > 0.05 else 'Medium' if imp > 0.02 else 'Low' 
                             for imp in importances]
            })
            
            # Sort by importance
            word_df = word_df.sort_values('Importance', ascending=False)
            
            # Display top influential words
            st.write("**Top 10 Most Influential Words:**")
            st.dataframe(word_df.head(10), use_container_width=True)
            
            # Classify words by sentiment type
            positive_words, negative_words, neutral_words = classify_words_by_sentiment(words, importances)
            
            st.subheader(" Words by Sentiment Category")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(" **Positive Words:**")
                if positive_words:
                    for word, importance in sorted(positive_words, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {word} (influence: {importance:.3f})")
                else:
                    st.write("No strong positive words detected")
            
            with col2:
                st.write(" **Negative Words:**")
                if negative_words:
                    for word, importance in sorted(negative_words, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {word} (influence: {importance:.3f})")
                else:
                    st.write("No strong negative words detected")
            
            with col3:
                st.write(" **Neutral/Other Words:**")
                if neutral_words:
                    top_neutral = sorted(neutral_words, key=lambda x: x[1], reverse=True)[:5]
                    for word, importance in top_neutral:
                        st.write(f"• {word} (influence: {importance:.3f})")
                else:
                    st.write("No neutral words")
            
            # Summary insights
            st.subheader(" Analysis Summary")
            
            high_influence_words = [word for word, imp in zip(words, importances) if imp > 0.05]
            avg_importance = np.mean(importances) if importances else 0
            
            insights = []
            if high_influence_words:
                insights.append(f" **Key words driving sentiment:** {', '.join(high_influence_words[:3])}")
            
            if positive_words and negative_words:
                insights.append("⚖ **Mixed sentiment detected** - text contains both positive and negative elements")
            elif positive_words:
                insights.append(" **Predominantly positive language** detected")
            elif negative_words:
                insights.append(" **Predominantly negative language** detected")
            
            insights.append(f" **Average word influence:** {avg_importance:.3f}")
            
            for insight in insights:
                st.write(insight)
        else:
            st.warning("Could not analyze word importance. Please try with a longer text.")

elif text == "":
    st.info(" Enter some text above to see detailed sentiment analysis with word breakdown!")

# Add some example texts
st.subheader(" Try these examples:")
examples = [
    "I love this product but the delivery was terrible!",
    "The movie was amazing and the acting was brilliant.",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special but not bad either."
]

for i, example in enumerate(examples):
    if st.button(f"Example {i+1}: {example[:50]}...", key=f"example_{i}"):
        st.text_input("Enter text for sentiment analysis:", value=example, key=f"input_{i}")
