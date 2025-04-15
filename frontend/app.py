import streamlit as st
import requests

# Streamlit page setup
st.set_page_config(page_title="Question Answering App", page_icon="❓")

st.title("💡 Smart Question Answering")
st.write("Ask a question and choose the type of response you'd like:")

# Input fields
question = st.text_input("Enter your question:")
chain_type = st.radio("Select Question Type:", options=["plain", "investment"])

# Submit button
if st.button("Submit Question"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        # Send request to your backend
        url = "http://localhost:5001/ask"
        headers = {"Content-Type": "application/json"}
        payload = {
            "question": question,
            "chain_type": chain_type
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"**Answer:** {data.get('answer')}")
                st.info(f"**Source:** {data.get('sources')}")
            else:
                st.error(f"Error: Received status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

# Footer
st.markdown("---")
st.caption("🚀 Powered by US at VIT haha.")
