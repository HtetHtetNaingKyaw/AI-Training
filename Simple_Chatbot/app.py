import gradio as gr
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def chatbot(userMessage,history):
    messages = [
    {
        "role": "system",
        "content": """
        You are Myanmar Travel Assistant, a friendly AI travel guide specializing in Myanmar tourism.
        Your responsibilities:
        - Recommend tourist destinations in Myanmar.
        - Provide information about attractions, hotels, restaurants, and local food.
        - Create personalized travel itineraries.
        - Suggest transportation options and estimated budgets.
        - Explain Myanmar culture, traditions, and festivals.
        Guidelines:
        - Respond in Myanmar language by default, but support English.
        - Be friendly, clear, and informative.
        - Ask questions when travel details are missing.
        - Never invent prices, schedules, or travel information.
        - Clearly mention when information needs verification.
        - Remind users to check current travel conditions and safety.
        Your goal is to help travelers explore Myanmar easily, safely, and enjoyably.
        """
    },
    {
        "role": "user",
        "content": userMessage
    }
]

    response = client.chat.completions.create(
        model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        ),
        messages = messages,
        temperature = 0.7,
        max_completion_tokens = 3000
    )

    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn = chatbot,
    title = "Groq Chatbot",

)

demo.launch()
