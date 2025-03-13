import torch
from transformers import pipeline
import pandas as pd
import yake
from keybert import KeyBERT
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import groq
import streamlit as st

# Streamlit UI Setup
st.title("📊 Sentiment Dashboard for Product Reviews")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file with tweets", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Load Zero-Shot Classification Model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Define candidate labels
    candidate_labels = ["negative", "positive", "neutral"]

    # Store results
    sentiment_counts = {"negative": 0, "positive": 0, "neutral": 0}
    sentiment_results = []

    for index, row in df.iterrows():
        text = row["tweet"]  
        
        # Predict sentiment
        result = classifier(text, candidate_labels)
        predicted_label = result["labels"][0]  

        # Store the result
        sentiment_results.append({"text": text, "sentiment": predicted_label})
        sentiment_counts[predicted_label] += 1

    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)

    # Display DataFrame
    st.subheader("🔍 Sentiment Analysis Results")
    st.write(sentiment_df)

    # Compute percentage sentiment perception
    total_reviews = len(sentiment_df)
    average_sentiment = {label: (count / total_reviews) * 100 for label, count in sentiment_counts.items()}

    # Extract positive and negative reviews
    positive_reviews = sentiment_df[sentiment_df["sentiment"] == "positive"]["text"].tolist()
    negative_reviews = sentiment_df[sentiment_df["sentiment"] == "negative"]["text"].tolist()

    # 🔥 Keyword Extraction using YAKE and KeyBERT
    kw_extractor = yake.KeywordExtractor(lan="en", n=5, top=20)
    keybert_model = KeyBERT()

    # Extract keywords from positive and negative reviews
    positive_keywords = kw_extractor.extract_keywords(" ".join(positive_reviews))
    negative_keywords = kw_extractor.extract_keywords(" ".join(negative_reviews))

    # Extract KeyBERT keywords (alternative)
    positive_keybert = keybert_model.extract_keywords(" ".join(positive_reviews), top_n=20)
    negative_keybert = keybert_model.extract_keywords(" ".join(negative_reviews), top_n=20)

    # Convert keywords to sets
    positive_words = {word[0] for word in positive_keywords + positive_keybert if len(word[0].split()) > 1}
    negative_words = {word[0] for word in negative_keywords + negative_keybert if len(word[0].split()) > 1}

    # Display Sentiment Stats
    st.subheader("📊 Sentiment Distribution")
    st.write(average_sentiment)

    # Display Keywords
    st.subheader("✅ Most Common Positive Words")
    st.write(positive_words)

    st.subheader("❌ Most Common Negative Words")
    st.write(negative_words)

    # Sentiment count visualization
    st.subheader("📉 Sentiment Pie Chart")
    fig, ax = plt.subplots()
    df_counts = sentiment_df["sentiment"].value_counts()
    ax.pie(df_counts, labels=df_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#f1c40f'])
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    # Word Cloud Visualization
    st.subheader("☁️ Word Clouds")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Positive Keywords")
        wordcloud_pos = WordCloud(width=400, height=200, background_color='white').generate(" ".join(positive_words))
        st.image(wordcloud_pos.to_array())

    with col2:
        st.subheader("❌ Negative Keywords")
        wordcloud_neg = WordCloud(width=400, height=200, background_color='white').generate(" ".join(negative_words))
        st.image(wordcloud_neg.to_array())

    # 🔥 Generate Summary using Groq API
    st.subheader("📢 Product Summary")
    api_key = st.text_input("Enter Groq API Key", type="password")
    product_name = st.text_input("Enter Product Name", value="iPhone")

    def generate_product_summary_groq(product_name, positive_keywords, negative_keywords, api_key):
        client = groq.Groq(api_key=api_key)

        prompt = f"""
        Generate a concise product summary based on the following information:

        Product Name: {product_name}

        Positive Keywords: {', '.join(positive_keywords)}
        Negative Keywords: {', '.join(negative_keywords)}

        Focus on highlighting the key strengths and weaknesses based on the provided keywords.
        """

        try:
            chat_completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",  
                messages=[{"role": "user", "content": prompt}]
            )

            return chat_completion.choices[0].message.content.strip()

        except groq.APIError as e:
            st.error(f"Groq API Error: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

    if st.button("Generate Summary"):
        if api_key:
            summary = generate_product_summary_groq(product_name, list(positive_words), list(negative_words), api_key)
            if summary:
                st.success("📢 Generated Content:")
                st.write(summary)
        else:
            st.error("Please enter a valid Groq API key.")
