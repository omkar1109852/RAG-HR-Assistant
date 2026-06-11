import json
import os
import sys
from dotenv import load_dotenv
import numpy as np
from typing import TypedDict
import math

#Data Ingestion
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from pdf2image import convert_from_path
from langchain_core.output_parsers import StrOutputParser
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

#Graph
from langgraph.graph import StateGraph
from langgraph.graph import END
from pydantic import BaseModel, Field
from typing import Literal

## Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\OmkarIngale\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

## Vector Embedding and vector store
load_dotenv()
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

##Create a state for Langgraph
class GraphState(TypedDict):
    question: str
    answer: str
    sources: list
    chat_history: list
    route: str

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

#Question with context to get confidence scores
question_generator = (contextualize_q_prompt | llm | StrOutputParser())

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

    # Get similarity scores
    standalone_question = question_generator.invoke({"input": query, "chat_history": chat_history})

    docs_and_scores = (vectorstore_faiss.similarity_search_with_score(standalone_question, k=3))

    source_details = []

    for doc, score in docs_and_scores:

        confidence = (math.exp(-score)* 100)
        confidence = round(confidence, 2)

        source_details.append(
            {
                "source": doc.metadata["filename"],
                "confidence": round(confidence, 2)
            }
        )

    return {
        "answer": response["answer"],
        "sources": source_details
    }
##Router Node

def router_node(state):

    prompt = f"""
You are an HR query router.

Classify the query into ONE category.

Return ONLY:

hr
casual

HR includes:
- leave policies
- parental leave
- caregiver leave
- holidays
- benefits
- bonus programs
- compensation
- eligibility
- workplace policies
- employee rules
- payroll
- PTO
- jury duty
- bereavement leave

Examples:

Q: What is bonus eligible earnings?
A: hr

Q: How many caregiver leave days do I get?
A: hr

Q: When does paid parental leave expire?
A: hr

Q: What is the time period within which I can use paid parental leave?
A: hr

Q: What holidays do we observe?
A: hr

Q: Hello
A: casual

Q: Thank you
A: casual

Q: What was the first question I asked?
A: casual

IMPORTANT:
If there is ANY possibility that a query relates to an employee policy,
benefit, compensation program, leave program, workplace rule, payroll,
holiday, or HR document, classify it as hr.

When uncertain, choose hr.

User Question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "route": response.content.strip().lower()
    }

##Create nodes
def hr_node(state):
    faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    result = get_response_llm(
        faiss_index,
        state["question"],
        state["chat_history"]
    )

    st.session_state.last_sources = result["sources"]

    return {
        **state,
        "answer": result["answer"],
        "sources": result["sources"]
    }

def casual_node(state):

    messages = []

    for msg in state["chat_history"]:

        if isinstance(msg, HumanMessage):

            messages.append(
                {
                    "role": "user",
                    "content": msg.content
                }
            )

        elif isinstance(msg, AIMessage):

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content
                }
            )

    messages.append(
        {
            "role": "user",
            "content": state["question"]
        }
    )

    response = llm.invoke(messages)

    return {
        **state,
        "answer": response.content,
        "sources": []
    }

##Routing Function
def route_decision(state):

    if state["route"] == "hr":
        return "hr_node"

    return "casual_node"



def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with HR documents")

    if ("chat_history" not in st.session_state):
        st.session_state.chat_history = []

    if ("last_sources" not in st.session_state):
        st.session_state.last_sources = []

    user_question = st.text_input("Ask a Question from the Files")
    
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)

    graph.add_node("hr_node", hr_node)

    graph.add_node("casual_node", casual_node)
    
    graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "hr_node": "hr_node",
        "casual_node": "casual_node"
    }
    )

    graph.add_edge("hr_node", END)

    graph.add_edge("casual_node", END)

    graph.set_entry_point("router")

    app = graph.compile()

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                final_documents = data_ingestion('documents')
                get_vector_store(final_documents)
                st.success("Done")

    if st.button("LLM Output"):
        st.session_state.last_sources = []
        with st.spinner("Processing..."):
            result = app.invoke(
            {
                "question": user_question,
                "chat_history": st.session_state.chat_history
            }
            )
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
            if result["sources"]:

                st.subheader("Sources")

                for item in result["sources"]:
                    st.write(f"📄 {item['source']}")
                    st.write(f"Relevance Score: {item['confidence']}%")

            st.success("Done")

if __name__ == "__main__":
    main()
