import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/query/"

# Streamlit UI
st.title("📊 AI-Powered Dataset Query Chatbot")

st.markdown("Ask questions about your dataset and get instant responses!")

# Input for user query
user_query = st.text_input("Enter your query:", placeholder="e.g., Show me orderdates where SALES > 5000.")

if st.button("Submit Query"):
    if user_query:
        with st.spinner("Processing..."):
            response = requests.post(API_URL, json={"query": user_query})
            if response.status_code == 200:
                result = response.json()
                st.success("Query Result:")
                st.json(result)  # Display the response as JSON
            else:
                st.error("Failed to retrieve results. Please try again.")
    else:
        st.warning("Please enter a query.")

# Run with: streamlit run app.py
