import json
import os
import sys
from dotenv import load_dotenv
import numpy as np

#Data Ingestion
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

# Vector Embedding And Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# LLm Models
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq


#Frontend
import streamlit as st


## Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\OmkarIngale\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

## Data ingestion

def ocr_pdf(pdf_path):

    pages = convert_from_path(
        pdf_path,
        dpi=300,
        poppler_path=r"C:\poppler\poppler-26.02.0\Library\bin"
    )

    docs = []

    for page_num, page in enumerate(pages):

        text = pytesseract.image_to_string(
            page,
            config="--psm 6"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1
                }
            )
        )

    return docs

def data_ingestion(folder_path):
    documents = []

    for file_path in Path(folder_path).rglob("*"):
        if not file_path.is_file():
            continue

        try:
            if file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()

            elif file_path.suffix.lower() == ".pdf":
                if (
                    "success" in file_path.name.lower()
                    and "bonus" in file_path.name.lower()
                ):

                    print(
                        f"Using OCR for {file_path.name}"
                    )

                    docs = ocr_pdf(
                        str(file_path)
                    )

                else:

                    loader = PyPDFLoader(
                        str(file_path)
                    )

                    docs = loader.load()

            else:
                continue

            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = str(file_path)
                doc.metadata["filename"] = file_path.name

            documents.extend(docs)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Total documents loaded: {len(documents)}")
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    final_documents=text_splitter.split_documents(documents)
    
    return final_documents



## Vector Embedding and vector store
load_dotenv()
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

## llm
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

## RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
PROMPT = ChatPromptTemplate.from_messages([("human", message)])


def get_response_llm(vectorstore_faiss,prompt,query):
    retriever = vectorstore_faiss.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
    )
    rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm
    response=rag_chain.invoke(query)
    docs = retriever.invoke(query)
    sources = list({
    d.metadata['source']
    for d in docs
    })
    
    return {
        "answer": response.content,
        "sources": sources
    }

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with HR documents")

    user_question = st.text_input("Ask a Question from the Files")

    
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                final_documents = data_ingestion('documents')
                get_vector_store(final_documents)
                st.success("Done")

    if st.button("LLM Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            #faiss_index = get_vector_store(docs)
            result = get_response_llm(faiss_index, PROMPT, user_question)
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(f"📄 {source}")
            st.success("Done")

if __name__ == "__main__":
    main()
