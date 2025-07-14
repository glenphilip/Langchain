from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Langsmith Tracking (optional)
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")

## Prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the queries"),
        ("user","Question:{question}")
    ]
)

## Streamlit framework
st.title("Langchain Demo with Llama 3-8b")
input_text=st.text_input("Search the topic you want")

## Groq LLM with Llama 3-8b
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    temperature=0.7,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    try:
        response = chain.invoke({'question':input_text})
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure your GROQ_API_KEY is set in your .env file")