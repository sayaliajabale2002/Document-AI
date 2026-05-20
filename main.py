from langchain.chat_models import init_chat_model
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph,START,END
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st 
import shutil 

import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


st.title("RAG PROJECT")
load_dotenv()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

llm_model = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

class State(TypedDict):
    question: str 
    answer: str 
    context: str 

def retrive(state:State):
    context = st.session_state.vector_store.similarity_search(state["question"],k=2)
    return {"context": context}

def llm_ans(state:State):
    prompt = f'''you are an assistant who will assist user questions ${state["question"]} based on data ${state["context"]}'''
    answer = llm_model.invoke(prompt)
    return {"answer": answer}

graph = StateGraph(State)
graph.add_node("retrive",retrive)
graph.add_node("llm_ans",llm_ans)
graph.add_edge(START,"retrive")
graph.add_edge("retrive","llm_ans")
graph.add_edge("llm_ans",END)

agent = graph.compile()


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if st.button("upload"):
    if uploaded_file is not None:
        with open("temp.pdf","wb") as f: shutil.copyfileobj(uploaded_file,f)

        load_pdf = PyPDFLoader("temp.pdf")
        embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        text_split_logic = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        text_split = text_split_logic.split_documents(load_pdf.load())
        st.session_state.vector_store = Chroma.from_documents(text_split,embedding=embedding)
        st.write("file uploaded successfully")


is_disabled = st.session_state.vector_store is None

user_input = st.text_input("Enter you query...",disabled=is_disabled)
send_button = st.button("send",disabled=is_disabled)
if send_button:
    if st.session_state.vector_store is not None:
        if user_input:
            que = {"question": user_input}
            ans = agent.invoke(que)
            st.write(ans["answer"].content)  