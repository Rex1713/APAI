import pandas as pd
import torch
import nltk
from collections import Counter
from transformers import pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv("iphone_tweets.csv")

# Load BART Sentiment Model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["negative", "neutral", "positive"]

# Sentiment Analysis
sentiment_scores = []
processed_tweets = []

for tweet in df["tweet"]:
    result = classifier(tweet, candidate_labels)
    predicted_label = result["labels"][0]
    sentiment_scores.append(predicted_label)
    processed_tweets.append(tweet.lower())  # Store processed tweets

# Store predictions
df["predicted_sentiment"] = sentiment_scores

# Compute sentiment distribution
sentiment_counts = df["predicted_sentiment"].value_counts(normalize=True) * 100
average_sentiment = sentiment_counts.to_dict()

# Process text for feature extraction
df["clean_tweet"] = df["tweet"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Function to get important words per sentiment
def get_top_words(sentiment, n=10):
    subset = df[df["predicted_sentiment"] == sentiment]["clean_tweet"]
    
    if subset.empty:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    tfidf_matrix = vectorizer.fit_transform(subset)
    feature_names = vectorizer.get_feature_names_out()
    
    word_scores = tfidf_matrix.sum(axis=0).A1  # Sum TF-IDF scores for each word
    word_ranking = sorted(zip(feature_names, word_scores), key=lambda x: x[1], reverse=True)

    return [word for word, score in word_ranking][:n]

# Get top words for positive and negative sentiments
top_positive_words = get_top_words("positive", n=10)
top_negative_words = get_top_words("negative", n=10)

# Generate Summary using BART Model
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")

summary_input = (
    f"Users have shared their opinions on the iPhone. "
    f"{average_sentiment.get('positive', 0):.1f}% of them are positive, appreciating aspects like {', '.join(top_positive_words)}. "
    f"However, {average_sentiment.get('negative', 0):.1f}% are negative, citing issues such as {', '.join(top_negative_words)}. "
    f"The remaining {average_sentiment.get('neutral', 0):.1f}% have neutral opinions."
)

summary = summarizer(summary_input, max_length=80, num_return_sequences=1)[0]['generated_text']

# Display Results
print("\n📊 Average Sentiment Perception:")
print(average_sentiment)

print("\n✅ Most Common Positive Words:")
print(top_positive_words)

print("\n❌ Most Common Negative Words:")
print(top_negative_words)

print("\n📢 Summary:")
print(summary)
