import praw
import pandas as pd
from datetime import datetime, timezone
import torch
from transformers import pipeline
import yake
from keybert import KeyBERT
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import groq
import streamlit as st
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Streamlit UI Setup
st.title("📊 Sentiment Dashboard for Product Reviews (Reddit)")

# Product Search Input
search_query = st.text_input("Enter product name for Reddit scraping:", "Samsung S25 Ultra")

# Function to Fetch Reddit Posts
def fetch_reddit_data(query):
    CLIENT_ID = "2u51dvjjMh43MVlsCfLb_A"
    CLIENT_SECRET = "792P9oRfpGWe0RkCn7IUIq55cwZeUQ"
    USER_AGENT = "YourAppName v1.0"

    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    subreddit_name = "all"

    posts = []
    for post in reddit.subreddit(subreddit_name).search(query, limit=100):  
        post_datetime = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)  
        post_date = post_datetime.strftime('%Y-%m-%d')  
        post_time = post_datetime.strftime('%H:%M:%S')  

        text_content = post.selftext.strip() if post.selftext.strip() else post.title.strip()
        posts.append([post_date, post_time, text_content])

    df = pd.DataFrame(posts, columns=["date", "Time", "tweet"])
    df.to_csv("reddit_topic_data.csv", index=False)
    return df

# Fetch Reddit Data When Button Clicked
if st.button("Scrape Reddit Posts"):
    df = fetch_reddit_data(search_query)
    st.success(f"Data on '{search_query}' fetched and saved to reddit_topic_data.csv!")
else:
    df = pd.read_csv("reddit_topic_data.csv")

df["date"] = pd.to_datetime(df["date"])

# Load Zero-Shot Classification Model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["negative", "positive", "neutral"]

# Store results
sentiment_counts = {"negative": 0, "positive": 0, "neutral": 0}
sentiment_results = []

for index, row in df.iterrows():
    text = row["tweet"]
    result = classifier(text, candidate_labels)
    predicted_label = result["labels"][0]  

    sentiment_results.append({"text": text, "sentiment": predicted_label})
    sentiment_counts[predicted_label] += 1

# Compute Overall Sentiment Score
total_weighted_sentiment = (
    sentiment_counts["positive"] * 1 +
    sentiment_counts["neutral"] * 0 +
    sentiment_counts["negative"] * -1
)
total_reviews = len(df)
overall_sentiment_score = total_weighted_sentiment / total_reviews

# Display Overall Sentiment Score
st.subheader("📈 Overall Sentiment Score")
st.metric(label="Sentiment Score", value=f"{overall_sentiment_score:.2f}", delta=0)

if overall_sentiment_score > 0.2:
    sentiment_status = "Mostly Positive 😀"
elif overall_sentiment_score < -0.2:
    sentiment_status = "Mostly Negative 😞"
else:
    sentiment_status = "Neutral 😐"

st.write(f"**Overall Sentiment:** {sentiment_status}")

# Convert to DataFrame
sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df["date"] = df["date"]

st.subheader("🔍 Sentiment Analysis Results")
st.write(sentiment_df)

# 📅 Sentiment Trend Over Time
st.subheader("📅 Sentiment Trend Over Time")
sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
sentiment_df["sentiment_score"] = sentiment_df["sentiment"].map(sentiment_mapping)

sentiment_trend = sentiment_df.groupby("date")["sentiment_score"].mean()

fig, ax = plt.subplots(figsize=(10, 4))
sentiment_trend.plot(ax=ax, marker="o", linestyle="-", color="b")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Date")
ax.set_ylabel("Average Sentiment Score")
ax.set_title("Sentiment Trend Over Time")
plt.xticks(rotation=45)
st.pyplot(fig)

# Weekly Sentiment Trend
sentiment_df["Week"] = sentiment_df["date"].dt.to_period("W").astype(str)
weekly_trend = sentiment_df.groupby("date")["sentiment_score"].mean()

st.subheader("📅 Sentiment Trend Over Weeks")
st.line_chart(weekly_trend)

# Compute percentage sentiment perception
average_sentiment = {label: (count / total_reviews) * 100 for label, count in sentiment_counts.items()}

# Extract positive and negative reviews
positive_reviews = sentiment_df[sentiment_df["sentiment"] == "positive"]["text"].tolist()
negative_reviews = sentiment_df[sentiment_df["sentiment"] == "negative"]["text"].tolist()

# 🔥 Keyword Extraction using YAKE and KeyBERT
kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=20)
keybert_model = KeyBERT()

# Remove product name from reviews for better keyword extraction
# clean_reviews = lambda reviews: [
#     re.sub(rf"\b{re.escape(search_query)}\b", "", review, flags=re.IGNORECASE) 
#     for review in reviews
# ]

# cleaned_positive = clean_reviews(positive_reviews)
# cleaned_negative = clean_reviews(negative_reviews)

# positive_keywords = kw_extractor.extract_keywords(" ".join(cleaned_positive))
# negative_keywords = kw_extractor.extract_keywords(" ".join(cleaned_negative))

# positive_keybert = keybert_model.extract_keywords(" ".join(cleaned_positive), top_n=20)
# negative_keybert = keybert_model.extract_keywords(" ".join(cleaned_negative), top_n=20)

# def extract_meaningful_keywords(keywords):
#     return {word[0] for word in keywords if len(word[0].split()) > 1 and word[0] not in ENGLISH_STOP_WORDS}

# positive_words = extract_meaningful_keywords(positive_keywords + positive_keybert)
# negative_words = extract_meaningful_keywords(negative_keywords + negative_keybert)

# st.subheader("📊 Sentiment Distribution")
# st.write(average_sentiment)

# st.subheader("✅ Most Common Positive Words")
# st.write(positive_words)

# st.subheader("❌ Most Common Negative Words")
# st.write(negative_words)

# Define list of brand names to exclude dynamically
brand_names = {"samsung", "apple", "google", "sony", "nokia", "huawei", "xiaomi", "oneplus", "motorola", "oppo", "vivo"}

# Function to clean reviews and remove brand names
# Convert search query into a set of words to exclude
def clean_reviews(reviews, product_name):
    words_to_exclude = brand_names.union(set(product_name.lower().split()))
    return [
        " ".join([word for word in review.split() if word.lower() not in words_to_exclude])
        for review in reviews
    ]

# Apply cleaning to positive and negative reviews
cleaned_positive = clean_reviews(positive_reviews, search_query)
cleaned_negative = clean_reviews(negative_reviews, search_query)

# Extract keywords using YAKE
positive_keywords = kw_extractor.extract_keywords(" ".join(cleaned_positive))
negative_keywords = kw_extractor.extract_keywords(" ".join(cleaned_negative))

# Extract keywords using KeyBERT
positive_keybert = keybert_model.extract_keywords(" ".join(cleaned_positive), top_n=20)
negative_keybert = keybert_model.extract_keywords(" ".join(cleaned_negative), top_n=20)

# Function to filter meaningful keywords (excluding brand names and product words)
def extract_meaningful_keywords(keywords, product_name):
    exclude_words = brand_names.union(set(product_name.lower().split()))
    
    # Check if any word in the keyword phrase belongs to excluded words
    def is_relevant(phrase):
        words = phrase.lower().split()
        return not any(word in exclude_words for word in words)

    return {word[0] for word in keywords if len(word[0].split()) > 1 and is_relevant(word[0])}

# Apply stronger filtering
positive_words = extract_meaningful_keywords(positive_keywords + positive_keybert, search_query)
negative_words = extract_meaningful_keywords(negative_keywords + negative_keybert, search_query)

# Display results in Streamlit
st.subheader("✅ Most Common Positive Keywords (Without Brand/Product Names)")
st.write(positive_words)

st.subheader("❌ Most Common Negative Keywords (Without Brand/Product Names)")
st.write(negative_words)
def find_sentences_with_keywords(reviews, keywords, max_sentences=3):
        """
        Extracts example sentences from reviews that contain the given keywords.
        Limits to 'max_sentences' per keyword.
        """
        keyword_sentences = {}
        for keyword in keywords:
            matched_sentences = [review for review in reviews if keyword.lower() in review.lower()]
            keyword_sentences[keyword] = matched_sentences[:max_sentences]  # Limit number of examples
        return keyword_sentences

    # Extract context examples
positive_examples = find_sentences_with_keywords(positive_reviews, positive_words)
negative_examples = find_sentences_with_keywords(negative_reviews, negative_words)

    # 🔹 Display Keyword Explanation & Context
    # 🔹 Display Keyword Explanation & Context
st.subheader("📖 Keyword Explanation & Context")

col1, col2 = st.columns(2)

with col1:
        st.subheader("✅ Positive Keyword Examples")
        for keyword, sentences in positive_examples.items():
            if sentences:
                with st.expander(f"**{keyword}**"):
                    for sentence in sentences:
                        highlighted_sentence = sentence.replace(keyword, f"**{keyword}**")
                        st.write(f"- {highlighted_sentence}")

with col2:
        st.subheader("❌ Negative Keyword Examples")
        for keyword, sentences in negative_examples.items():
            if sentences:
                with st.expander(f"**{keyword}**"):
                    for sentence in sentences:
                        highlighted_sentence = sentence.replace(keyword, f"**{keyword}**")
                        st.write(f"- {highlighted_sentence}")



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

def generate_product_summary_groq(product_name, positive_keywords, negative_keywords, api_key):
    client = groq.Groq(api_key=api_key)

    prompt = f"""
    Generate a concise product summary based on the following:
    Product Name: {product_name}
    Positive Keywords: {', '.join(positive_keywords)}
    Negative Keywords: {', '.join(negative_keywords)}
Focus on highlighting the key strengths and weaknesses based on the provided keywords in bullet points.
    """

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

if st.button("Generate Summary"):
    if api_key:
        summary = generate_product_summary_groq(search_query, list(positive_words), list(negative_words), api_key)
        if summary:
            st.success("📢 Generated Content:")
            st.write(summary)
    else:
        st.error("Enter a valid Groq API key.")
