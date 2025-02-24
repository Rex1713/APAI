import torch
from transformers import pipeline
from datasets import load_dataset
from collections import Counter
import pandas as pd
import re

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
    text = row["tweet"]
    
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

def extract_keywords(text_list, num_keywords=10):
    words = " ".join(text_list).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', words)  # Extract only words
    common_words = Counter(words).most_common(num_keywords)
    return {word[0] for word in common_words}

# Get most common words
positive_words = extract_keywords(positive_reviews, 10)
negative_words = extract_keywords(negative_reviews, 10)

# Prepare input for summarization
summary_input = (
    f"Users have shared their opinions on the iPhone. "
    f"Positive feedback highlights aspects such as {', '.join(positive_words)}. "
    f"Negative feedback often mentions concerns about {', '.join(negative_words)}."
)

# Generate summary
summary = summarizer(summary_input, max_length=80, min_length=30, do_sample=False)[0]['summary_text']

# Print results
print("\U0001F4CA Average Sentiment Perception:", average_sentiment)
print("\u2705 Most Common Positive Words:", positive_words)
print("\u274C Most Common Negative Words:", negative_words)
print("\U0001F4E2 Summary:")
print(summary)
