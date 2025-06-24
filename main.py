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
st.write("**Most Frequent Words:**")
    freq_df = pd.DataFrame(most_common, columns=['Word', 'Count'])
    st.dataframe(freq_df, use_container_width=True)
    
    # Sentiment Analysis
    st.subheader("🎭 Sentiment Analysis")
    
    clean_text = preprocess_text(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = F.softmax(outputs.logits, dim=-1)
    labels = ['Negative', 'Neutral', 'Positive']
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = predictions[0][predicted_class].item()
    
    # Display sentiment result
    sentiment_color = {"Negative": "🔴", "Neutral": "🟡", "Positive": "🟢"}
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
    st.subheader("🔍 Word Influence Analysis")
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
            
            st.subheader("📝 Words by Sentiment Category")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("🟢 **Positive Words:**")
                if positive_words:
                    for word, importance in sorted(positive_words, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {word} (influence: {importance:.3f})")
                else:
                    st.write("No strong positive words detected")
            
            with col2:
                st.write("🔴 **Negative Words:**")
                if negative_words:
                    for word, importance in sorted(negative_words, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {word} (influence: {importance:.3f})")
                else:
                    st.write("No strong negative words detected")
            
            with col3:
                st.write("🟡 **Neutral/Other Words:**")
                if neutral_words:
                    top_neutral = sorted(neutral_words, key=lambda x: x[1], reverse=True)[:5]
                    for word, importance in top_neutral:
                        st.write(f"• {word} (influence: {importance:.3f})")
                else:
                    st.write("No neutral words")
            
            # Summary insights
            st.subheader("💡 Analysis Summary")
            
            high_influence_words = [word for word, imp in zip(words, importances) if imp > 0.05]
            avg_importance = np.mean(importances) if importances else 0
            
            insights = []
            if high_influence_words:
                insights.append(f"🎯 **Key words driving sentiment:** {', '.join(high_influence_words[:3])}")
            
            if positive_words and negative_words:
                insights.append("⚖️ **Mixed sentiment detected** - text contains both positive and negative elements")
            elif positive_words:
                insights.append("😊 **Predominantly positive language** detected")
            elif negative_words:
                insights.append("😔 **Predominantly negative language** detected")
            
            insights.append(f"📈 **Average word influence:** {avg_importance:.3f}")
            
            for insight in insights:
                st.write(insight)
        else:
            st.warning("Could not analyze word importance. Please try with a longer text.")

elif text == "":
    st.info("👆 Enter some text above to see detailed sentiment analysis with word breakdown!")

# Add some example texts
st.subheader("💡 Try these examples:")
examples = [
    "I love this product but the delivery was terrible!",
    "The movie was amazing and the acting was brilliant.",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special but not bad either."
]

for i, example in enumerate(examples):
    if st.button(f"Example {i+1}: {example[:50]}...", key=f"example_{i}"):
        st.rerun()
