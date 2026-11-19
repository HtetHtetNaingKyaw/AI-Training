
import gradio as gr
import joblib
import re

# =========================
# 1. LOAD TRAINED MODELS
# =========================

sentiment_model = joblib.load("sentiment_model.joblib")
category_model = joblib.load("category_model.joblib")

print("Models loaded successfully!")


def sentimentPrediction(feedback):
    prediction = sentiment_model.predict([feedback])
    return prediction

def categoryPrediction(feedback):
    prediction = category_model.predict([feedback])
    return prediction

def prediction(feedback):
    sentiment = sentimentPrediction(feedback)
    category = categoryPrediction(feedback)
    return sentiment, category

demo = gr.Interface(
    fn = prediction,
    inputs = gr.TextArea(
        label="Enter Student Feedback",
        placeholder="Type student feedback here...",
        lines=6
    ),
    outputs =[gr.Textbox(label="Predicted Sentiment"), gr.Textbox(label="Predicted Category")],
    title="Student Sentiment & Category Prediction",
)

demo.launch()