import streamlit as st
import openai
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "My ollama Q&A chatbo
# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful friend. Please respond to the user."),
    ("user", "Question: {question}")
])
def gen_response(question, engine, temperature, max_token):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Basic Q&A Chatbot with Ollama")

st.sidebar.title("Settings")
engine = st.sidebar.selectbox("Select the OpenAI Model", ['gemma:2b'])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
max_token = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=100)

st.write("Let's go, ask me anything!")
query = st.text_input("You:")

if query :
    res = gen_response(query,engine, temperature, max_token)
    st.write(res)
else:
    st.write("Looks like you haven't asked anything yet!")
