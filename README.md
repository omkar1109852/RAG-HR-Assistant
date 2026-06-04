# HR Document Q&A Assistant

## Prerequisites

* Python 3.10+
* OpenAI API Key
* Groq API Key

---

## Setup


### 1. Create and Activate a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Add Documents

Place all HR documents inside the `documents/` folder.

Supported formats:

* `.pdf`
* `.docx`

Example:

```text
documents/
├── Leave_Policy.pdf
├── Benefits.docx
└── Employee_Handbook.pdf
```

---

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

---

## Create the Vector Store

1. Open the application in the browser.
2. Click **Vectors Update**.
3. Wait until processing is complete.

This will:

* Load all documents
* Split documents into chunks
* Generate embeddings
* Create and save the FAISS index

---

## Ask Questions

1. Enter a question in the text box.
2. Click **LLM Output**.
3. The application will return:

   * Answer generated from the HR documents
   * Source document(s) used for retrieval

Example questions:

* How many annual leave days do employees receive?
* What is the parental leave policy?
* Are contractors eligible for company-authorized holidays?

---

## Notes

* Re-run **Vectors Update** whenever documents are added, removed, or modified.
* The FAISS index is stored locally in the `faiss_index/` directory.
* Answers are generated only from retrieved document content.

---

# Decisions and Assumptions Log
# Phase - 1
## Project Goal

Build a Retrieval-Augmented Generation (RAG) HR Assistant capable of answering questions from a collection of HR policy documents (PDF and DOCX formats).

---

# Decisions

## 1. Document Loading Strategy

### Decision
- Use `Docx2txtLoader` for `.docx` files
- Use `PyPDFLoader` for `.pdf` files

### Reasoning
- Native LangChain support
- Simple implementation
- Supports the document formats provided in the dataset

### Assumption
- All HR documents are available locally in the `documents/` directory

---

## 2. Recursive Folder Traversal

### Decision
Use:

```python
Path(folder_path).rglob("*")
```

### Reasoning
- Automatically discovers all supported files
- Supports nested folder structures
- Eliminates the need to manually specify file paths

---

## 3. Chunking Strategy

### Decision

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

### Reasoning
- Preserves semantic structure better than fixed-size splitting
- Reduces information loss near chunk boundaries
- Serves as a strong baseline for future evaluation

### Future Work
- Compare against alternative chunk sizes
- Evaluate retrieval quality using different chunking strategies

---

## 4. Embedding Model

### Decision

```python
OpenAIEmbeddings(
    model="text-embedding-3-large"
)
```

### Reasoning
- Strong semantic retrieval performance
- Easy integration with LangChain
- Widely used in production RAG systems

### Tradeoff
- Requires API usage and associated cost

---

## 5. Vector Store

### Decision

```python
FAISS
```

### Reasoning
- Runs locally
- No additional infrastructure required
- Suitable for the current document corpus size

### Alternatives Considered
- Chroma
- Pinecone

### Why FAISS Was Chosen
- Simpler setup
- No dependency on external services
- Adequate for the expected dataset size

---

## 6. Retrieval Strategy

### Decision

```python
search_type="similarity"
k=3
```

### Reasoning
- Easy to understand and debug
- Good baseline for evaluation
- Sufficient for the initial implementation

### Future Work
- Experiment with different values of K
- Evaluate MMR retrieval

---

## 7. LLM Selection

### Decision

```python
llama-3.1-8b-instant
```

via Groq

### Reasoning
- Fast inference
- Cost-effective
- Good instruction-following capability

---

## 8. Prompt Design

### Decision

The model is instructed to answer only using retrieved context.

### Prompt Principle

> Answer this question using the provided context only.

### Reasoning
- Reduces hallucinations
- Improves answer grounding
- Aligns with RAG best practices

---

## 9. Source Attribution

### Decision

Return source filenames alongside generated answers.

### Reasoning
- Improves transparency
- Makes responses verifiable
- Helps evaluate retrieval quality

---

## 10. User Interface

### Decision

```python
Streamlit
```

### Reasoning
- Fast prototyping
- Easy demonstration
- Minimal frontend development effort

---

# Known Limitations

- No conversation memory
- No chunking evaluation framework
- No retrieval accuracy metrics
- No reranking
- No hybrid search
- No agentic workflow

---

# Planned Phase 2 Work

## Objectives

- Create an evaluation dataset (10–15 representative HR questions)
- Measure retrieval accuracy using Recall@K
- Compare chunking strategies
- Recommend the best chunking strategy based on empirical results

---

# Planned Phase 3 Work

## Objectives

- Add conversational memory
- Support follow-up questions
- Improve multi-turn interactions

---

# Planned Phase 4 Work (Optional)

## Objectives

- Introduce LangGraph orchestration
- Add HR query classification
- Route between retrieval and general conversation flows

---

# Notes

This document is intended to be a living engineering log and will be updated throughout the project to capture implementation decisions, assumptions, tradeoffs, and lessons learned.
