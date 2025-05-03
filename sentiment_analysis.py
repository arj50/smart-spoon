import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Path to your survey data
FILE_PATH = 'data/survey_data.csv'

# 1. Load the data

df = pd.read_csv(FILE_PATH)

# 2. Rename the columns to match keys (positional, in case of spaces)
cols = df.columns.tolist()
rename_map = {
    cols[13]: 'Concerns',
    cols[14]: 'Suggestions'
}
df = df.rename(columns=rename_map)

# 3. Combine text fields for analysis
text_data = pd.concat([df['Concerns'].fillna(''), df['Suggestions'].fillna('')])

# 4. Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# 5. Compute sentiment scores
sentiments = text_data.apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# 6. Classify sentiment
def label_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

labels = sentiments.apply(label_sentiment)

# 7. Summary of sentiment distribution
sentiment_counts = labels.value_counts(normalize=True) * 100
print("Sentiment Distribution (%):")
print(sentiment_counts.round(2), "\n")

# 8. Plot sentiment distribution
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['#99ff99','#ff9999','#cccccc'])
plt.title('Overall Sentiment of Responses')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 9. Extract top keywords for negative and positive groups
vectorizer = CountVectorizer(stop_words='english', max_features=20)

# Positive responses
pos_texts = text_data[labels == 'Positive']
pos_vec = vectorizer.fit_transform(pos_texts)
pos_freq = pd.DataFrame({ 'word': vectorizer.get_feature_names_out(), 'count': pos_vec.sum(axis=0).A1 })
pos_top = pos_freq.sort_values('count', ascending=False).head(10)

# Negative responses
neg_texts = text_data[labels == 'Negative']
neg_vec = vectorizer.fit_transform(neg_texts)
neg_freq = pd.DataFrame({ 'word': vectorizer.get_feature_names_out(), 'count': neg_vec.sum(axis=0).A1 })
neg_top = neg_freq.sort_values('count', ascending=False).head(10)

print("Top Keywords in Positive Responses:\n", pos_top.to_string(index=False), "\n")
print("Top Keywords in Negative Responses:\n", neg_top.to_string(index=False), "\n")

# 10. Plot top keywords
plt.figure(figsize=(8,4))
plt.barh(pos_top['word'], pos_top['count'], color='#99ff99')
plt.title('Top Keywords (Positive)')
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.barh(neg_top['word'], neg_top['count'], color='#ff9999')
plt.title('Top Keywords (Negative)')
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()
