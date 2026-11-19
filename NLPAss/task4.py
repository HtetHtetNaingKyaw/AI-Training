import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dataset = pd.read_csv('student_feedback.csv')
text_data = dataset['feedback_text'].iloc[:15].tolist()  # Example: select first 5 texts
print("Text data:", text_data)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(text_data)

print("Embedding shape:", embeddings.shape)
print(embeddings)

# Calculate cosine similarity between embeddings
score = cosine_similarity([embeddings[0]], [embeddings[1]])
print("Cosine similarity:", score)  # Compare the first text with the rest}