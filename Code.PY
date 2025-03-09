import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
file_path = "/content/imdb movie dataset.csv"
df = pd.read_csv(file_path)

# Drop missing plot values
df = df.dropna(subset=['plot'])

# Sentiment analysis function
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['Sentiment'] = df['plot'].apply(get_sentiment)

# Display sentiment distribution
sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Analysis of IMDb Movie Plots")
plt.show()

# Display a few results
print(df[['title', 'plot', 'Sentiment']].head())
