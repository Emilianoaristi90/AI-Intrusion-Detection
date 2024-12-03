
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Page Configuration
st.set_page_config(page_title="Power BI Style Intrusion Dashboard", layout="wide")
st.title("Intrusion Detection Dashboard - Power BI Style")
st.write("Analyze, classify, and monitor network traffic with an interactive, visually appealing dashboard.")

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
        st.header("Key Metrics")
        total_records = len(data)
        malicious_count = data[label_column].value_counts().get(1, 0)
        normal_count = total_records - malicious_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total_records)
        col2.metric("Malicious Records", malicious_count)
        col3.metric("Normal Records", normal_count)

        # Label Distribution
        st.header("Label Distribution")
        label_dist = data[label_column].value_counts()
        pie_fig = px.pie(
            values=label_dist,
            names=label_dist.index,
            title="Distribution of Labels",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # Feature Distribution
        st.header("Feature Analysis")
        selected_feature = st.selectbox("Select Feature for Analysis", data.columns[:-1])
        hist_fig = px.histogram(
            data, x=selected_feature,
            title=f"Distribution of {selected_feature}",
            color=label_column,
            marginal="box",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        # Train-Test Split and Model Training
        X = data.drop(columns=[label_column], errors='ignore')
        y = data[label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Classification Report
        st.header("Classification Report")
        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        bar_fig = px.bar(
            report_df.iloc[:-1, :],
            x=report_df.index[:-1],
            y="f1-score",
            title="F1-Score by Class",
            color="f1-score",
            color_continuous_scale=px.colors.sequential.Teal
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Confusion Matrix
        st.header("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Normal", "Malicious"], y=["Normal", "Malicious"],
            colorscale="Blues", showscale=True
        ))
        cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(cm_fig, use_container_width=True)

else:
    st.info("Please upload a dataset to start.")
