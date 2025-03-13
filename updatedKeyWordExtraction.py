import torch
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import yake
from keybert import KeyBERT
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
kw_extractor = yake.KeywordExtractor(lan="en", n=5, top=20)
keybert_model = KeyBERT()

# Extract keywords from positive and negative reviews
positive_keywords = kw_extractor.extract_keywords(" ".join(positive_reviews))
negative_keywords = kw_extractor.extract_keywords(" ".join(negative_reviews))

# Extract KeyBERT keywords (alternative)
positive_keybert = keybert_model.extract_keywords(" ".join(positive_reviews), top_n=20)
negative_keybert = keybert_model.extract_keywords(" ".join(negative_reviews), top_n=20)

# Convert keywords to sets
positive_words = {word[0] for word in positive_keywords + positive_keybert if len(word[0].split()) > 1 and "iphone" not in word[0].lower()}
negative_words = {word[0] for word in negative_keywords + negative_keybert if len(word[0].split()) > 1}

# # 🔥 Prepare input for content generation with structured insights
# summary_input = (
#     f"Users have shared their opinions on the iPhone, highlighting various aspects. "
#     f"Positive feedback frequently emphasizes {', '.join(positive_words)}, showcasing what users love about the device. "
#     f"On the other hand, concerns have been raised regarding {', '.join(negative_words)}, reflecting areas where users feel improvements are needed. "
#     f"Based on these insights, generate an engaging and informative piece that captures the essence of user sentiment, weaving together both praises and critiques into a well-rounded narrative."
# )

# # Generate content
# generated_content = summarizer(summary_input, max_length=100, min_length=40, do_sample=False)[0]['summary_text']
# 🔥 Summarize the dataset first to get the main themes
full_text = " ".join(df["tweet"].tolist())  # Merge all tweets
dataset_summary = summarizer(full_text, max_length=150, min_length=80, do_sample=False)[0]['summary_text']

# 🔥 Construct a better input for summarization
summary_input = (
    f"Users have shared their opinions on the iPhone, highlighting various aspects. "
    f"From a broad analysis of tweets, we observe key sentiments: {dataset_summary} "
    f"Positive feedback frequently focuses on features such as {', '.join(positive_words)}, highlighting what users love about the device. "
    f"However, users have raised concerns about {', '.join(negative_words)}, indicating areas that need improvement. "
    f"Based on these insights, generate a well-structured summary that blends both positive and negative feedback in a natural, readable manner."
)

# 🔥 Generate refined summary
generated_content = summarizer(summary_input, max_length=150, min_length=80, do_sample=False)[0]['summary_text']

# Print results
print("📢 Generated Content:")
print(generated_content)




# Print results
print("📊 Average Sentiment Perception:", average_sentiment)
print("✅ Most Common Positive Words:", positive_words)
print("❌ Most Common Negative Words:", negative_words)
print("📢 Generated Content:")
print(generated_content)

# Sentiment count visualization
df_counts = sentiment_df["sentiment"].value_counts()

fig, ax = plt.subplots()
ax.pie(df_counts, labels=df_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#f1c40f'])
ax.set_title("Sentiment Distribution")
plt.show()

# Word cloud visualization
positive_text = " ".join(positive_words)  # Use extracted keywords
negative_text = " ".join(negative_words)  # Use extracted keywords


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

wordcloud_pos = WordCloud(width=400, height=200, background_color='white').generate(positive_text)
ax1.imshow(wordcloud_pos, interpolation='bilinear')
ax1.set_title("✅ Positive Keywords")
ax1.axis("off")

wordcloud_neg = WordCloud(width=400, height=200, background_color='white').generate(negative_text)
ax2.imshow(wordcloud_neg, interpolation='bilinear')
ax2.set_title("❌ Negative Keywords")
ax2.axis("off")

plt.show()
