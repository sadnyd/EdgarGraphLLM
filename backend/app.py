import os
import warnings
import textwrap
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain and Neo4j imports
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain

# Ignore warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv('EDGAR_URI')
NEO4J_USERNAME = os.getenv('EDGAR_USERNAME')
NEO4J_PASSWORD = os.getenv('EDGAR_PASSWORD')
NEO4J_DATABASE = os.getenv('EDGAR_DATABASE', 'neo4j') # Default to 'neo4j' if not set
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Check if essential variables are loaded
if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_KEY]):
    raise ValueError("Missing essential environment variables (Neo4j URI/Username/Password, Gemini API Key)")

# Global constants from the notebook
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding' # Make sure this matches your index creation

# --- Initialize LangChain Components (Done once on startup) ---

# Initialize Embeddings and LLM
print("Initializing Embeddings and LLM...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Use gemini-1.5-flash as gemini-2.0-flash might not be available via API key yet
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        convert_system_message_to_human=True # Often needed for Gemini models in Langchain
    )
    print("Embeddings and LLM initialized successfully.")
except Exception as e:
    print(f"Error initializing Google AI components: {e}")
    raise

# Initialize Neo4j Vector Stores and Retrievers
print("Initializing Neo4j Vector Stores...")
try:
    # Plain vector store
    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        index_name=VECTOR_INDEX_NAME,
        text_node_property=VECTOR_SOURCE_PROPERTY, # Property containing the text
        embedding_node_property=VECTOR_EMBEDDING_PROPERTY # Property containing the embedding
    )
    retriever = vector_store.as_retriever()
    print("Plain vector store initialized.")

    # Investment-enhanced retrieval query
    investment_retrieval_query = """
    MATCH (node)-[:PART_OF]->(f:Form),
        (f)<-[:FILED]-(com:Company),
        (com)<-[owns:OWNS_STOCK_IN]-(mgr:Manager)
    WITH node, score, mgr, owns, com
        ORDER BY owns.shares DESC LIMIT 10
    WITH collect (
        mgr.managerName +
        " owns " + owns.shares +
        " shares in " + com.companyName +
        " at a value of $" +
        apoc.number.format(toInteger(owns.value)) + "."
    ) AS investment_statements, node, score
    RETURN apoc.text.join(investment_statements, "\n") +
        "\n" + node.text AS text, // Combine investment info and original chunk text
        score,
        {
          source: node.source, // Ensure 'source' property exists on Chunk node
          chunkId: node.chunkId // Add chunkId if useful for debugging
        } as metadata
    """

    vector_store_with_investment = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        index_name=VECTOR_INDEX_NAME,
        retrieval_query=investment_retrieval_query,
        # text_node_property is implicitly handled by the RETURN statement in retrieval_query
    )
    retriever_with_investments = vector_store_with_investment.as_retriever()
    print("Investment-enhanced vector store initialized.")

except Exception as e:
    print(f"Error initializing Neo4j Vector Stores: {e}")
    print("Please ensure the Neo4j instance is running, credentials are correct,")
    print(f"and the vector index '{VECTOR_INDEX_NAME}' exists with the specified properties.")
    raise

# Create QA Chains
print("Creating QA Chains...")
try:
    plain_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff", # Use "stuff" for simplicity if context fits
        retriever=retriever,
        return_source_documents=True # Good practice to see what was retrieved
    )

    investment_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_with_investments,
        return_source_documents=True
    )
    print("QA Chains created successfully.")
except Exception as e:
    print(f"Error creating QA Chains: {e}")
    raise

# --- Flask App ---
app = Flask(__name__)
CORS(app) 
@app.route('/')
def home():
    return "QA API is running. Use the /ask endpoint."

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Endpoint to ask a question.
    Expects JSON payload:
    {
        "question": "Your question here",
        "chain_type": "plain" or "investment" (defaults to "plain")
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get('question')
    chain_type = data.get('chain_type', 'plain').lower() # Default to 'plain'

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    print(f"\nReceived question: '{question}' using chain: '{chain_type}'")

    try:
        if chain_type == 'investment':
            print("Using investment chain...")
            result = investment_chain(
                {"question": question},
                # return_only_outputs=True # Keep False to see source docs if needed
            )
        elif chain_type == 'plain':
            print("Using plain chain...")
            result = plain_chain(
                {"question": question},
                # return_only_outputs=True
            )
        else:
            return jsonify({"error": "Invalid 'chain_type'. Use 'plain' or 'investment'."}), 400

        print(f"LLM Answer: {result.get('answer', 'N/A')}")
        # print(f"Sources: {result.get('sources', 'N/A')}") # You might want to log/check sources

        # Return only the answer and sources as per the original notebook examples
        # The chain returns more, like 'source_documents'
        response_data = {
            "answer": result.get("answer", "No answer generated."),
            "sources": result.get("sources", ""), # The 'sources' key specifically
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing question: {e}")
        # Log the full traceback for debugging if needed
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred processing your request."}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    # Use host='0.0.0.0' to make it accessible on your network
    # Use debug=True for development (auto-reloads), False for production
    app.run(host='0.0.0.0', port=5001, debug=True)
    # For local testing only use app.run(port=5001, debug=True)
