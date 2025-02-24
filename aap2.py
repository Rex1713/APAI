import torch
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import yake
from keybert import KeyBERT
from collections import Counter

# Load Zero-Shot Classification Model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load Dataset
df = pd.read_csv("iphone_tweets.csv")

# Define candidate labels
candidate_labels = ["negative", "positive", "neutral"]

# Store results
sentiment_counts = {"negative": 0, "positive": 0, "neutral": 0}
sentiment_results = []

for index, row in df.iterrows():
    text = row["tweet"]  # Ensure you are using the correct column name
    
    # Predict sentiment
    result = classifier(text, candidate_labels)
    predicted_label = result["labels"][0]  # Get the highest-scoring label
    
    # Store the result
    sentiment_results.append({"text": text, "sentiment": predicted_label})
    sentiment_counts[predicted_label] += 1

# Convert to DataFrame
sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df.to_csv("iphone_sentiment_results.csv", index=False)

# Compute percentage sentiment perception
total_reviews = len(sentiment_df)
average_sentiment = {label: (count / total_reviews) * 100 for label, count in sentiment_counts.items()}

# Extract positive and negative reviews
positive_reviews = sentiment_df[sentiment_df["sentiment"] == "positive"]["text"].tolist()
negative_reviews = sentiment_df[sentiment_df["sentiment"] == "negative"]["text"].tolist()

# 🔥 Better Keyword Extraction using YAKE and KeyBERT
kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=10)
keybert_model = KeyBERT()

# Extract keywords from positive and negative reviews
positive_keywords = kw_extractor.extract_keywords(" ".join(positive_reviews))
negative_keywords = kw_extractor.extract_keywords(" ".join(negative_reviews))

# Extract KeyBERT keywords (alternative)
positive_keybert = keybert_model.extract_keywords(" ".join(positive_reviews), top_n=10)
negative_keybert = keybert_model.extract_keywords(" ".join(negative_reviews), top_n=10)

# Convert keywords to sets
positive_words = {word[0] for word in positive_keywords + positive_keybert}
negative_words = {word[0] for word in negative_keywords + negative_keybert}

# 🔥 Prepare input for summarization with improved structure
summary_input = (
    f"Users have shared their opinions on the iPhone. "
    f"Most positive feedback focuses on {', '.join(positive_words)}. "
    f"However, negative reviews often mention concerns about {', '.join(negative_words)}."
)

# Generate summary
summary = summarizer(summary_input, max_length=100, min_length=40, do_sample=False)[0]['summary_text']

# Print results
print("📊 Average Sentiment Perception:", average_sentiment)
print("✅ Most Common Positive Words:", positive_words)
print("❌ Most Common Negative Words:", negative_words)
print("📢 Summary:")
print(summary)
