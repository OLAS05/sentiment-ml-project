import pandas as pd
import streamlit as st
import os

# --- Step 1: Load predictions CSV ---
predictions_file = "output/predictions.csv"

if not os.path.exists(predictions_file):
    st.error(f"Predictions file not found at {predictions_file}. Please run run_inference.py first.")
else:
    df = pd.read_csv(predictions_file)

    # --- Step 2: Page title ---
    st.title("Support Tickets Sentiment Dashboard")

    # --- Step 3: Show table of tickets ---
    st.subheader("All Tickets")
    st.dataframe(df)

    # --- Step 4: Sentiment distribution chart ---
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # --- Step 5: Filter tickets by sentiment ---
    st.subheader("Filter Tickets")
    selected_sentiment = st.selectbox(
        "Select sentiment to view",
        options=["All"] + df["sentiment"].unique().tolist()
    )

    if selected_sentiment != "All":
        filtered_df = df[df["sentiment"] == selected_sentiment]
    else:
        filtered_df = df

    # Show filtered tickets
    st.dataframe(filtered_df)