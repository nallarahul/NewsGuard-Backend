import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the trained model and tokenizer from the local directory
MODEL_PATH = "final_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Save model and tokenizer as a pickle file
import pickle

with open("model.pkl", "wb") as model_file:
    pickle.dump({"model": model.state_dict(), "tokenizer": tokenizer}, model_file)

print("Model saved as model.pkl")