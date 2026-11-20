import chromadb
from dotenv import load_dotenv
import os
from groq import Groq
from chromadb.utils import embedding_functions
import gradio as gr

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-Multilingual-MiniLM-L12-v2"
)

db = chromadb.PersistentClient(
    path = "./chroma_db"
)

collection = db.get_or_create_collection(
    name = "my_knowledge",
    embedding_function = ef,
    metadata = {
        "hnsw:space" : "cosine"
    }
)

if collection.count() == 0:
    with open(
    "knowledge.txt",
    "r",
    encoding="utf-8"
    ) as f:
        text = f.read()

    chunks = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    collection.add(
        documents = chunks,
        ids = [ f"doc - {i}"
               for i in range (len(chunks))]
    )

###Ends of KB

def chatbot(userMessage,history):

    results = collection.query(
        query_texts = [userMessage],
        n_results = min(3,collection.count())
    )

    documents = results["documents"][0]
    context = "\n\n".join(documents)

    messages = [
        {
            "role": "system",
            "content": f"""
            You are a helpful AI assistant.
            Answer using the following context.
            If the answer is not available
            in the context, say that the
            information is not available
            in the knowledge base.
            Context:{context}
             """
        },
        {
            "role": "user",
            "content": userMessage
        }
    ]


    response = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = messages,
        max_completion_tokens = 3000,
        temperature = 0.7
    )

    return response.choices[0].message.content
##End chatbot function

demo = gr.ChatInterface(
    fn = chatbot,
    title = "RAG Chatbot",
    description="Ask question from KB"
)

demo.launch()

