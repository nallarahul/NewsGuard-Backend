import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Define dataset directory
dataset_dir = "Newsguard AI\FakeNewsNet\dataset"

def clean_text(text):
    """Removes special characters, extra spaces, and converts text to lowercase."""
    text = re.sub(r'[^a-zA-Z0-9 ]', '', str(text))  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

def load_csv_data():
    """Loads and preprocesses CSV files from FakeNewsNet dataset."""
    files = ["gossipcop_fake.csv", "gossipcop_real.csv", "politifact_fake.csv", "politifact_real.csv"]
    all_data = []
    
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "title" in df.columns:  # Using 'title' as the text field
                df = df[["title"]].dropna()
                df.rename(columns={"title": "text"}, inplace=True)
                
                # Assign labels manually based on dataset type
                if "gossipcop" in file:
                    df["label"] = 1  # Assume GossipCop news is fake
                elif "politifact" in file:
                    df["label"] = 0  # Assume PolitiFact news is real
                
                df["text"] = df["text"].apply(clean_text)
                all_data.append(df)
        else:
            print(f"Warning: {file_path} not found.")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Load and split data
df = load_csv_data()
if df.empty:
    print("Error: No valid data loaded. Check dataset structure.")
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("Newsguard AI/backend/train.csv", index=False)
    test_df.to_csv("Newsguard AI/backend/test.csv", index=False)
    print("Preprocessing complete. Train and test datasets saved in backend folder.")
