from sklearn.feature_extraction.text import TfidfVectorizer

documents  = [
    'AI improves earning',
    'Ai improves business'
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())