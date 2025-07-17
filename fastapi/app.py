from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version='1.0',
    description="A simple API server"
)

model = ChatGroq(model="llama3-70b-8192")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

add_routes(app, prompt1 | model, path="/essay")
add_routes(app, prompt2 | model, path="/poem")

llm_local = Ollama(model="llama2")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
