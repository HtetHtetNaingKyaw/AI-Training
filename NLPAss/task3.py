import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

datast = pd.read_csv('student_feedback.csv')
text_data = datast['feedback_text']
#count vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)
print(vectorizer.get_feature_names_out())
print(X.toarray())

#tfidf vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)
print(vectorizer.get_feature_names_out())
print(X.toarray())
