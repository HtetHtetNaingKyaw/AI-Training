import pandas as pd
import re

dataset = pd.read_csv('student_feedback.csv')
text_data = dataset['feedback_text']

for sentence in text_data:
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\s++', ' ', sentence).strip()
    tokens = sentence.split()
    print(tokens)


