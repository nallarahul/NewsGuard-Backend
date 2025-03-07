from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load the trained model and tokenizer from Hugging Face Model Hub
MODEL_NAME = "nallarahul/NewsGaurd"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define labels
labels = ["Fake", "Real"]

@app.route('/')
def hello_world():
    return "<p>hello world</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get prediction
    predicted_label = labels[torch.argmax(probs).item()]
    confidence = {labels[i]: round(probs[0][i].item() * 100, 2) for i in range(len(labels))}

    return jsonify({"prediction": predicted_label, "confidence": confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
