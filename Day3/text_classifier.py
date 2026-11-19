from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts = [    
"I love this product",    
"The service is excellent",    
"This is a wonderful experience",    
"I am very happy with the purchase",    
"The product is fantastic",    
"I really like this service",    
"I hate this product",    
"The service is terrible",    
"This is a bad experience",    
"I am very disappointed",    
"The product is awful",    
"I do not like this service"
]

labels = [    
"Positive", 
"Positive", 
"Positive",    
"Positive", 
"Positive", 
"Positive",    
"Negative", 
"Negative", 
"Negative",    
"Negative", 
"Negative", 
"Negative"
]

X_train,X_test,y_train,y_test =train_test_split(texts,labels,test_size=0.25)

model  = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X_train,y_train)

test_prediction = model.predict(X_test)
train_prediction = model.predict(X_train)

print(test_prediction)
print(y_test)

print(train_prediction)
print(y_train)

test_accuracy = accuracy_score(y_test, test_prediction)
train_accuracy = accuracy_score(y_train, train_prediction)

print('Testing Accuracy: ', test_accuracy)
print('Training Accuracy: ', train_accuracy)

new_text = ['This product is terrible']
print(model.predict(new_text))