import streamlit as st
import requests

# Streamlit page setup
st.set_page_config(page_title="EDGAR Q&A Assistant", page_icon="📊", layout="centered")

# --- HEADER SECTION ---
st.markdown("""
    <h1 style='text-align: center;'>📄 EDGAR-Based Question Answering Bot</h1>
    <p style='text-align: center; font-size: 18px;'>Get answers directly from official SEC filings (10-K, 13F, etc.)</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- PAGE SELECTION ---
page = st.radio("Choose Mode", ["Ask a Question", "Upload PDF"])

if page == "Ask a Question":
    # --- INSTRUCTIONS for Asking a Question ---
    with st.expander("📘 How to Ask a Question"):
        st.markdown("""
        **Welcome to the EDGAR Q&A Assistant!** This tool allows you to ask questions about public companies using official filings from the **EDGAR database**.

        **Types of Questions:**
        - **Plain**: Based on standard filings such as **Form 10-K**, 10-Q, and others.
        - **Investment**: Extracted from institutional investment data such as **Form 13F**.

        **How to Use:**
        1. Enter your question (e.g., *"What are Apple's revenue sources?"*).
        2. Choose the context of your question — *Plain* or *Investment*.
        3. Click **Submit** to receive an answer based on SEC documents.

        📌 **Note**: This tool uses AI to extract relevant answers from real financial documents.
        You’ll also get the source document(s) that support the response.

        ⚠️ **Disclaimer**: This is an educational tool. Please consult a financial advisor before making any investment decisions.
        """)

    # --- MAIN INTERACTION SECTION for Asking a Question ---
    st.subheader("💬 Ask Your Question")

    question = st.text_input("What would you like to know about a company?", placeholder="e.g., What is Tesla's debt situation?")
    chain_type = st.radio("Choose the context of your question:", options=["plain", "investment"])

    if st.button("📤 Submit Question"):
        if not question.strip():
            st.warning("⚠️ Please enter a question first.")
        else:
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
                    st.success("✅ Answer retrieved!")
                    st.markdown(f"### 🧠 Answer\n**{data.get('answer')}**")
                    st.markdown(f"### 📁 Source Document(s)\n{data.get('sources')}")
                else:
                    st.error(f"❌ Server returned status code {response.status_code}")
            except Exception as e:
                st.error(f"❌ Could not connect to backend: {e}")

elif page == "Upload PDF":
    # --- INSTRUCTIONS for Uploading PDF ---
    with st.expander("📘 How to Upload a PDF and Ask a Question"):
        st.markdown("""
        **Welcome to the PDF Question Assistant!** This tool allows you to upload a PDF document and ask questions about its content.

        **How to Use:**
        1. Upload your PDF file (e.g., an SEC filing or other documents).
        2. Enter your question about the PDF content.
        3. Click **Submit** to receive an answer based on the content of the PDF.

        📌 **Note**: This tool uses AI to analyze the text in your PDF and answer your questions based on the content.

        ⚠️ **Disclaimer**: The tool analyzes the text within the PDF but may not be perfect. Always review your PDF content for accuracy.
        """)

    # --- MAIN INTERACTION SECTION for Uploading PDF ---
    st.subheader("📄 Upload PDF and Ask Your Question")

    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    question_pdf = st.text_input("What would you like to know about the PDF?", placeholder="e.g., What is the key financial ratio in the report?")

    if st.button("📤 Submit PDF Question"):
        if pdf_file is None or not question_pdf.strip():
            st.warning("⚠️ Please upload a PDF file and enter a question.")
        else:
            files = {'pdf': pdf_file}
            data = {'question': question_pdf}
            try:
                response = requests.post("http://localhost:5001/ask-pdf", files=files, data=data)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Answer retrieved!")
                    st.markdown(f"### 🧠 Answer\n**{data.get('answer')}**")
                else:
                    st.error(f"❌ Server returned status code {response.status_code}")
            except Exception as e:
                st.error(f"❌ Could not connect to backend: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("🛠️ Built by us at VIT :) | EDGAR Q&A Assistant")
