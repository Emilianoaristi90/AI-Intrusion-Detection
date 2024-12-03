
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff
import smtplib
from email.mime.text import MIMEText

# Page Configuration
st.set_page_config(page_title="Enhanced Intrusion Detection Dashboard", layout="wide")
st.title("Enhanced Intrusion Detection Dashboard")
st.write("Analyze, classify, and monitor network traffic with advanced features.")

# Sidebar: File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset Preview:")
    st.sidebar.dataframe(data.head())

    # Detect or select label column
    potential_labels = ['Label', 'Target', 'Class', 'attack_cat']
    label_column = None
    for col in potential_labels:
        if col in data.columns:
            label_column = col
            break

    if label_column is None:
        st.sidebar.error("No label column detected! Please select one.")
        label_column = st.sidebar.selectbox("Select the label column:", data.columns)
    else:
        st.sidebar.success(f"Using '{label_column}' as the label column.")

    if label_column:
        # Dataset Summary
        st.header("Dataset Summary")
        st.write(f"**Total Records**: {len(data)}")
        st.write("**Column Information**:")
        st.write(data.describe())

        # Visualize Label Distribution
        st.header("Label Distribution")
        label_dist = data[label_column].value_counts()
        fig = px.pie(values=label_dist, names=label_dist.index, title="Label Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Advanced Visualization: Feature Distribution
        st.header("Feature Distribution")
        feature = st.selectbox("Select a feature to visualize:", data.columns[:-1])
        hist_fig = px.histogram(data, x=feature, title=f"Distribution of {feature}", marginal="box")
        st.plotly_chart(hist_fig, use_container_width=True)

        # Train-Test Split and Model Training
        X = data.drop(columns=[label_column], errors='ignore')
        y = data[label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Show Results
        st.header("Model Predictions")
        st.write("Classification Report:")
        report = classification_report(y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.header("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        cm_fig = ff.create_annotated_heatmap(
            z=cm, x=list(set(y)), y=list(set(y)), colorscale="Viridis"
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        # Alert System
        st.header("Alert System")
        threshold = st.slider("Set Alert Threshold (% malicious traffic):", 0, 100, 30)
        malicious_rate = (predictions.sum() / len(predictions)) * 100
        if malicious_rate > threshold:
            st.warning(f"Alert! Malicious traffic detected: {malicious_rate:.2f}% exceeds threshold of {threshold}%")
            if st.button("Send Alert Notification"):
                try:
                    # Sending Email Alert (example setup)
                    sender_email = "your_email@example.com"
                    recipient_email = "recipient@example.com"
                    msg = MIMEText(f"Malicious traffic detected: {malicious_rate:.2f}% exceeds threshold.")
                    msg['Subject'] = "Intrusion Detection Alert"
                    msg['From'] = sender_email
                    msg['To'] = recipient_email
                    with smtplib.SMTP("smtp.example.com", 587) as server:
                        server.starttls()
                        server.login(sender_email, "your_password")
                        server.sendmail(sender_email, recipient_email, msg.as_string())
                    st.success("Alert notification sent!")
                except Exception as e:
                    st.error(f"Failed to send alert notification: {e}")
else:
    st.info("Please upload a dataset to start.")
