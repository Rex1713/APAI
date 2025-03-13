import torch
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import yake
from keybert import KeyBERT
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import groq

# Load Zero-Shot Classification Model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
    predicted_label = result["labels"][0]  

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
positive_words = {word[0] for word in positive_keywords + positive_keybert if len(word[0].split()) > 1 and "iphone" not in word[0].lower()}
negative_words = {word[0] for word in negative_keywords + negative_keybert if len(word[0].split()) > 1}

print("📊 Average Sentiment Perception:", average_sentiment)
print("✅ Most Common Positive Words:", positive_words)
print("❌ Most Common Negative Words:", negative_words)
print("📢 Generated Content:")
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
        print(f"Groq API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Example usage:
api_key = "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"  # Replace with your actual API key
product_name = "iphone"

summary = generate_product_summary_groq(product_name, list(positive_words), list(negative_words), api_key)

if summary:
    print("📢 Generated Content:")
    print(summary)


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

# 🔥 Generate Summary using Groq API
