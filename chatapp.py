import os
import streamlit as st
#from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Set up Streamlit
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Uploaded PDFs and chat with their content")

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Load documents from the specified directory
# documents = SimpleDirectoryReader('data').load_data()
# index = VectorStoreIndex.from_documents(documents)

# # Set up the retriever and postprocessor
# retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
# postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)

# either way we can now query the index
query_engine = index.as_query_engine()

# Create the query engine
# query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# Accept user input
user_input = st.chat_input("Type your message here...")

# Check if there is any input and that it's a valid string
if user_input and isinstance(user_input, str):
    # Save the input in a string variable
    saved_input = user_input

    # Display the saved input
    st.write("You entered:", saved_input)

    try:
        # Process the user input with the query engine
        response = query_engine.query(user_input)
        
        # Display the response
        st.write(f"Final Response: {response}")
        
        # Print the response with source
        # pprint_response(response, show_source=True)
    except Exception as e:
        st.error(f"An error occurred while processing your query: {e}")
else:
    st.warning("Please enter a valid string for the query.")
