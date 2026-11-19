import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

dataset = pd.read_csv('student_feedback.csv')
text = dataset['feedback_text']
sentiment_labels = dataset['sentiment']
category_labels = dataset['category']

# for sentiment
X_sentiment_train,X_sentiment_test,y_sentiment_train,y_sentiment_test =train_test_split(text,sentiment_labels,test_size=0.25)

sentiment_model  = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

sentiment_model.fit(X_sentiment_train,y_sentiment_train)
joblib.dump(
    sentiment_model,
    "sentiment_model.joblib"
)

print("Sentiment model trained and saved successfully.")

# for category
X_category_train,X_category_test,y_category_train,y_category_test =train_test_split(text,category_labels,test_size=0.25)

category_model  = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

category_model.fit(X_category_train,y_category_train)
joblib.dump(
    category_model,
    "category_model.joblib"
)
print("Category model trained and saved successfully.")
