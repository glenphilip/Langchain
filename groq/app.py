import streamlit as st
import os
import time
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Streamlit UI
st.title("🎥 YouTube Video Assistant (RAG Powered)")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
""")

# Function to get transcript
def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to create vector DB from transcript
def embed_transcript(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.create_documents([text])
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# YouTube URL input
video_url = st.text_input("📎 Enter YouTube Video URL")

if st.button("🔍 Process Video"):
    with st.spinner("Extracting transcript and building vector store..."):
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            st.session_state.vectors = embed_transcript(transcript_text)
            st.success("✅ Transcript processed and stored in vector DB!")

# User question
query = st.text_input("💬 Ask a question about the video:")

if query and "vectors" in st.session_state:
    with st.spinner("Retrieving and generating answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": query})
        elapsed_time = time.process_time() - start_time
        
        st.write("🧠 **Answer:**")
        st.write(response["answer"])
        st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

        with st.expander("📄 Relevant Transcript Sections"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("---")
