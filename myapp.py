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
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
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

#Chat History
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)


## Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\OmkarIngale\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

## Vector Embedding and vector store
load_dotenv()
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

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
    #text_splitter = SemanticChunker(embeddings)
    final_documents=text_splitter.split_documents(documents)
    return final_documents




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

contextualize_q_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given a chat history and the latest user question,
                formulate a standalone question which can be
                understood without the chat history.

                Do not answer the question.
                """
            ),
            MessagesPlaceholder(
                "chat_history"
            ),
            (
                "human",
                "{input}"
            )
        ]
    )
)

qa_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an HR assistant.

                Answer ONLY using the provided context.

                If the answer is not found,
                explicitly say so.

                Context:
                {context}
                """
            ),
            MessagesPlaceholder(
                "chat_history"
            ),
            (
                "human",
                "{input}"
            )
        ]
    )
)

def get_response_llm(vectorstore_faiss, query, chat_history):

    retriever = (
        vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    )

    history_aware_retriever = (
        create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )
    )

    question_answer_chain = (
        create_stuff_documents_chain(
            llm,
            qa_prompt
        )
    )

    rag_chain = (
        create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
    )

    response = rag_chain.invoke(
        {
            "input": query,
            "chat_history": chat_history
        }
    )

    sources = list(
        {
            doc.metadata["filename"]
            for doc in response["context"]
        }
    )

    return {
        "answer": response["answer"],
        "sources": sources
    }

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with HR documents")

    if ("chat_history" not in st.session_state):
        st.session_state.chat_history = []

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
            result = get_response_llm(faiss_index, user_question, st.session_state.chat_history)
            st.session_state.chat_history.extend(
            [
            HumanMessage(content=user_question),
            AIMessage(content=result["answer"])
            ]
            )

            MAX_HISTORY = 10

            st.session_state.chat_history = (st.session_state.chat_history[-MAX_HISTORY:])

            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(f"📄 {source}")
            st.success("Done")

if __name__ == "__main__":
    main()