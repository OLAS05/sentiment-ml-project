import pandas as pd
from transformers import pipeline
import os
#print('I am gonna complete this project with the help of my GOD, JESUS Christ')
#print(df.head())
#Show column names
# print('\n columns in csv')
# print(df.columns)

# --- Step 1: Read the CSV file ---
df = pd.read_csv("data/tickets.csv")  # relative to project root

# Make sure 'text' column exists
if "message" not in df.columns:
    raise ValueError("CSV file must have a 'text' column")

# --- Step 2: Load sentiment model ---
sentiment_model = pipeline('sentiment-analysis')

# --- Step 3: Run sentiment analysis on all tickets
result = sentiment_model(df["message"].tolist())

# Add predictions to dataframe
df["sentiment"]= [r["label"] for r in result]
df["confidence"]= [r["score"] for r in result]

# --- Step 4: Create output folder if it doesn't exist ---
if not os.path.exists("output"):
    os.makedirs("output")

# --- Step 5: Save results ---
output_path = "output/predictions.csv"
df.to_csv(output_path, index=False)

print(f"✅ Sentiment analysis completed! Output saved to: {output_path}")

