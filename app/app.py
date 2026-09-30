from pathlib import Path
import streamlit as st
import joblib

# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model and TF-IDF vectorizer
model = joblib.load(BASE_DIR / "model" / "customer_query_svm.pkl")
vectorizer = joblib.load(BASE_DIR / "model" / "tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Query Routing",
    page_icon="💬",
    layout="centered"
)

# Title
st.title("💬 Customer Query Routing System")

st.write(
    "Enter a customer query below and the AI model will "
    "route it to the most relevant customer-service category."
)

# User input
query = st.text_area(
    "Enter your customer query:",
    placeholder="Example: My card has not arrived yet..."
)

# Prediction
if st.button("Route Query"):
    if query.strip() == "":
        st.warning("Please enter a customer query.")
    else:
        query_tfidf = vectorizer.transform([query])
        prediction = model.predict(query_tfidf)[0]

        st.success("Query routed successfully!")

        st.subheader("Predicted Category")
        st.write(prediction)