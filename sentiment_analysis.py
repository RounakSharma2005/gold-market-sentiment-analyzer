import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import yfinance as yf
import pandas as pd

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Fetch news from Google RSS

def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    news_items = feed.entries[:num_articles]

    articles = []
    for item in news_items:
        title = item.title
        link = item.link
        published = item.get("published", "N/A")
        content = fetch_article_content(link)

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles

# Extract article text

def fetch_article_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text() for p in paragraphs)

        return content.strip()

    except Exception:
        return ""

# Build Vector Store (RAG)

def build_vector_store(articles):
    texts = [article["title"] for article in articles]

    embeddings = embedding_model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts
def retrieve_context(query, index, texts, k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Retrieve similar context (RAG)

def retrieve_context(query, index, texts, k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    retrieved_texts = [texts[i] for i in indices[0]]
    return retrieved_texts

# Simulated Social Media Sentiment

def analyze_social_sentiment(query, index, texts):
    social_text = f"""
    Traders discussing {query} on social media.
    Opinions include fear, optimism, and speculation.
    """
    score, sentiment, explanation = analyze_sentiment(
        social_text, index, texts
    )
    return score, sentiment, explanation

# Sentiment Analysis 

def analyze_sentiment(text, index, texts):
    if not text.strip():
        return 0.0, "Neutral", "No text provided"

    context = retrieve_context(text, index, texts)

    context_block = "\n".join(context)

    prompt = f"""
    You are a financial market analyst.

    Use the following related past news headlines as context:
    {context_block}

    Now analyze the sentiment of the current headline.
    Classify it strictly as:
    Positive, Negative, or Neutral.

    Also provide a one-line explanation.

    Current headline: "{text}"
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    output = response.choices[0].message.content.strip()

    if "Positive" in output:
        sentiment = "Positive"
        score = 1.0
    elif "Negative" in output:
        sentiment = "Negative"
        score = -1.0
    else:
        sentiment = "Neutral"
        score = 0.0

    return score, sentiment, output

# Aggregate Multi-Source Sentiment

def aggregate_sentiment(news_score, social_score):

    news_weight = 0.7
    social_weight = 0.3
    final_score = (news_weight * news_score) + (social_weight * social_score)
    if final_score > 0.2:
        return "Positive", final_score
    elif final_score < -0.2:
        return "Negative", final_score
    else:
        return "Neutral", final_score

# Summary of sentiments
def summarize_sentiments(all_articles, index, texts):
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for article in all_articles:
        _, sentiment, _ = analyze_sentiment(
            article["title"], index, texts
        )
        summary[sentiment] += 1

    total = len(all_articles)
    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total}")

    for sentiment, count in summary.items():
        percent = (count / total) * 100 if total else 0
        print(f"{sentiment}: {count} ({percent:.2f}%)")

def get_actual_market_movement():
    """
    Uses real gold price data (XAUUSD) from Yahoo Finance.
    Returns: Positive / Negative / Neutral
    """
    try:
        gold = yf.download("XAUUSD=X", period="2d", interval="1d", progress=False)

        if len(gold) < 2:
            return "Neutral"

        yesterday_close = gold["Close"].iloc[-2]
        today_close = gold["Close"].iloc[-1]

        change_pct = ((today_close - yesterday_close) / yesterday_close) * 100

        if change_pct > 0.1:
            return "Positive"
        elif change_pct < -0.1:
            return "Negative"
        else:
            return "Neutral"

    except Exception as e:
        print("Gold price fetch failed:", e)
        return "Neutral"
    
prediction_history = []
def evaluate_prediction(predicted, actual):
    record = {
        "predicted": predicted,
        "actual": actual,
        "correct": predicted == actual
    }
    prediction_history.append(record)
    return record["correct"]

# Main function
def main():
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold analysis",
        "gold forecast",
        "gold investment"
    ]

    num_articles_per_query = 5
    all_articles = []
  
    for query in queries:
        print(f"\nFetching articles for: {query}")
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)
   
    index, texts = build_vector_store(all_articles)
 
    for idx, article in enumerate(all_articles, 1):

        
        score, sentiment, explanation = analyze_sentiment(
            article["title"], index, texts
        )
    
        actual_movement = get_actual_market_movement()
    
        is_correct = evaluate_prediction(
            predicted=sentiment,
            actual=actual_movement
        )

        print(f"\nArticle {idx}")
        print(f"Title: {article['title']}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Explanation: {explanation}")
        print(f"Actual Gold Market Movement: {actual_movement}")
        print(f"Prediction Correct: {is_correct}")

   
    summarize_sentiments(all_articles, index, texts)

def run_analysis_from_ui(queries, num_articles_per_query=5):
    all_articles = []

    for query in queries:
        articles = fetch_news(query, num_articles_per_query)
        all_articles.extend(articles)

    index, texts = build_vector_store(all_articles)

    results = []

    for article in all_articles:
        score, sentiment, explanation = analyze_sentiment(
            article["title"], index, texts
        )

        results.append({
            "title": article["title"],
            "sentiment": sentiment,
            "explanation": explanation
        })
    return results
# Entry Point
if __name__ == "__main__":
    main()
