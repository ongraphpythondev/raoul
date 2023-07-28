from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv
import streamlit as st
import pinecone 

load_dotenv()
st.title("Question Answer PDF")
# openai_api_key=os.getenv('openai_api_key')
# Pinecone_ENV=os.getenv("PINECONE_ENV")
# Pinecone_API=os.getenv("PINECONE_API_KEY")

Pinecone_ENV=st.secrets["Pinecone_ENV"]
openai_api_key=st.secrets["openai_api_key"]
Pinecone_API=st.secrets["PINECONE_API_KEY"]

# loader = DirectoryLoader('./data/', glob="**/*.pdf", show_progress=True)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


text=st.text_area("Text",height=300)
button=st.button("submit",use_container_width=True)

# initialize pinecone
pinecone.init(
    api_key=Pinecone_API,  # find at app.pinecone.io
    environment=Pinecone_ENV  # next to api key in console
)
index_name = "emails"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
if button:
#  if you already have an index, you can load it like this

# docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    
    docs = docsearch.similarity_search(text)
    st.write(docs[0].page_content)