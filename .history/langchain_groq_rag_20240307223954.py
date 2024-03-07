import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from video import TencentTTSClient


embed_model = 'text-embedding-ada-002'
openai_api_key = os.environ["OPENAI_API_KEY"]
secret_id = os.environ["VOICE_ID"]
secret_key = os.environ["VOICE_KEY"]

embed = OpenAIEmbeddings(
    model=embed_model,
    openai_api_key=openai_api_key
)

load_dotenv()  #

groq_api_key = os.environ['GROQ_API_KEY']


if "vector" not in st.session_state:
    st.session_state.embeddings = embed
    if 'loader' not in st.session_state:
        st.session_state.loader = UnstructuredMarkdownLoader("./chuyiAI.md")
    st.session_state.loader = UnstructuredMarkdownLoader("./chuyiAI.md")

    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs)
    st.session_state.vector = Chroma.from_documents(
        st.session_state.documents, st.session_state.embeddings)
    st.session_state.vectorstore_retreiver = st.session_state.vector.as_retriever(
        search_kwargs={"k": 3})
    st.session_state.keyword_retriever = BM25Retriever.from_documents(
        st.session_state.documents)
    st.session_state.keyword_retriever.k = 3
    st.session_state.ensemble_retriever = EnsembleRetriever(retrievers=[st.session_state.vectorstore_retreiver,
                                                                        st.session_state.keyword_retriever],
                                                            weights=[0.7, 0.3])
    st.session_state.ttsserver = TencentTTSClient(secret_id, secret_key)


st.title("ChuyiAI Chat ")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768'
)

prompt = ChatPromptTemplate.from_template("""
You are yimi, the AI intelligent assistant of ChuyiAI company.                                          
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer.
Please answer the question in Chinese.                                        
I will tip you $2000000 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.ensemble_retriever
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response["answer"])
    if len(response["answer"]) < 150:
        tts_file_path = st.session_state.ttsserver.text_to_voice(
            response["answer"])
        audio_file = open(tts_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # print(doc)
            # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content)
            st.write("--------------------------------")
