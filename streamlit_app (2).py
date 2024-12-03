
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Page Title
st.title("AI-Based Intrusion Detection System")
st.write("Upload a dataset to detect and classify network traffic anomalies.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Simulate model predictions (replace with your model)
    if 'Label' in data.columns:
        X = data.drop(columns=['Label'], errors='ignore')
        y = data['Label']

        # Simulated model (replace with your trained model)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)  # Train on uploaded data (for demo purposes)

        predictions = model.predict(X)
        data['Predicted_Label'] = predictions

        st.write("Prediction Results:")
        st.dataframe(data[['Label', 'Predicted_Label']].head())

        # Plot results
        st.write("Prediction Distribution:")
        fig, ax = plt.subplots()
        data['Predicted_Label'].value_counts().plot(kind='bar', ax=ax, title="Predicted Labels")
        st.pyplot(fig)
    else:
        st.error("The dataset must contain a 'Label' column for true values!")
else:
    st.info("Please upload a CSV file to proceed.")
