from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()


# Langsmith Tracking
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
st.title("Langchain Demo with Ollama Gemma2:2b")
input_text=st.text_input("Search the topic you want")

# Ollama Gemma2:2b
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    try:
        st.write(chain.invoke({"question":input_text}))
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure Ollama is running and the gemma2:2b model is installed. Run: ollama pull gemma2:2b")