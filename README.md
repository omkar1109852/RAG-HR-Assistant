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

# Phase - 1
## Project Goal

Build a Retrieval-Augmented Generation (RAG) HR Assistant capable of answering questions from a collection of HR policy documents (PDF and DOCX formats).

---

# Decisions

## 1. Document Loading Strategy

### Decision
- Use `Docx2txtLoader` for `.docx` files
- Use `PyPDFLoader` for `.pdf` files
- Use `pytesseract` for `.pdf` files which contained images

### Reasoning
- OCR reads content that cannot be read by usual pdf loaders
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
- Default chunking used in tutorial videos

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
- Easy integration with LangChain

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
- Cost-effective

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

## 11. OCR-Based Extraction for Success Bonus FAQ

### Problem Identified

During retrieval testing, questions related to the **2022 Success Bonus FAQ** consistently returned incorrect or incomplete answers despite the information being visibly present in the PDF document.

Examples included:

- What is bonus eligible earnings?
- Why are my earnings different than my current salary?

The information was present in the PDF but was not being retrieved by the RAG pipeline.

---

### Root Cause

Investigation showed that standard PDF extraction methods were unable to extract all content from the document.

The following approaches were tested:

- PyPDFLoader
- PyMuPDFLoader
- pymupdf4llm

While these methods successfully extracted portions of the document, several FAQ sections were embedded as image-based content and therefore unavailable to the vector store.

As a result, embeddings were generated from incomplete document text, causing retrieval failures.

---

### Decision

Implement OCR-based extraction for the **2022 Success Bonus FAQ (US).pdf** document only.

The following tools were used:

- pdf2image
- pytesseract

All other PDF documents continue to use the standard `PyPDFLoader` workflow.

---

### Reasoning

- OCR successfully extracted FAQ content that standard PDF parsers missed.
- The issue was isolated to a single document.
- Applying OCR selectively minimizes additional processing time.
- The approach avoids introducing OCR-related noise into documents that already have reliable text extraction.

---

### Implementation

1. Detect the Success Bonus FAQ document during ingestion.
2. Convert PDF pages to images using `pdf2image`.
3. Extract text from images using Tesseract OCR.
4. Convert OCR output into LangChain `Document` objects.
5. Apply the standard chunking, embedding, and indexing pipeline.

---

### Impact

- Improved retrieval quality for Success Bonus-related queries.
- Increased document coverage within the vector store.
- Reduced instances of missing context during retrieval.
- Enabled successful answering of previously failing FAQ questions.

---

### Tradeoffs

- OCR introduces additional processing time during ingestion.
- OCR may occasionally introduce character recognition errors.
- Additional dependencies are required:
  - Tesseract OCR
  - Poppler

---

### Key Learning

Document ingestion quality directly impacts retrieval performance. Before tuning embeddings, chunk sizes, or prompts, it is important to verify that source documents are being extracted correctly. In this case, the primary retrieval issue originated from incomplete PDF text extraction rather than the retrieval or generation components of the RAG pipeline.

---

## Phase 2 – Chunking Strategy Evaluation

### Objective

Evaluate the impact of different chunking strategies on retrieval quality for the HR Policy RAG Assistant. The goal was to determine which chunking approach provides the best balance between context preservation and retrieval accuracy.

### Chunking Strategies Evaluated

#### 1. Recursive Character Splitter (1000/100)
```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

#### 2. Recursive Character Splitter (500/50)
```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

#### 3. Recursive Character Splitter (1500/150)
```python
RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
```

#### 4. Character Splitter (1000/100)
```python
CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

#### 5. Semantic Chunker
```python
SemanticChunker(embeddings)
```

### Evaluation Methodology

- Evaluated all chunking strategies on the same corpus of HR policies, FAQs, and compensation documents.
- Created a benchmark of 30 representative HR-related questions covering:
  - Eligibility rules
  - Leave policies
  - Compensation and bonuses
  - Holiday policies
  - Bereavement and jury duty policies
  - Table-based and multi-condition questions
- Scoring rubric:
  - **1.0** = Complete and correct answer
  - **0.5** = Partially correct answer
  - **0.0** = Incorrect answer or retrieval failure

### Results

| Strategy | Score | Accuracy |
|-----------|--------|-----------|
| Recursive Character Splitter (1000/100) | 24.5 / 30 | 81.7% |
| Character Splitter (1000/100) | 22.5 / 30 | 75.0% |
| Recursive Character Splitter (500/50) | 19.5 / 30 | 65.0% |
| Semantic Chunker | 18.5 / 30 | 61.7% |
| Recursive Character Splitter (1500/150) | 18.0 / 30 | 60.0% |

### Key Findings

- **Recursive Character Splitter (1000/100)** achieved the best overall performance and provided the most consistent retrieval quality across different document types.
- Smaller chunks (500/50) often fragmented policy sections and eligibility tables, resulting in incomplete answers.
- Larger chunks (1500/150) preserved context but reduced retrieval precision.
- Semantic Chunking performed well on narrative content but struggled with structured HR documents containing tables, FAQs, and eligibility matrices.
- Character Splitter (1000/100) performed surprisingly well but was less reliable on multi-condition and long-context questions.

### Evaluation Artifacts

The detailed evaluation results used for chunking strategy comparison are available in:

`evaluation/chunking_strategy_evaluation.xlsx`

The spreadsheet contains:
- 30 benchmark questions
- Ideal answers
- Responses from each chunking strategy

### Conclusion

The **Recursive Character Splitter (1000/100)** was selected as the production chunking strategy because it achieved the highest overall retrieval accuracy while maintaining a good balance between context preservation and retrieval precision.


## Phase 3 – Conversational RAG

### Objective

Enhance the HR Assistant to support multi-turn conversations and follow-up questions.

### Implementation

Implemented conversational memory using:

- Streamlit Session State
- LangChain Chat History
- History-Aware Retriever
- Retrieval Chain

The chatbot maintains conversation context and reformulates follow-up questions into standalone queries before retrieval.

### Example

User:
What is bonus eligible earnings?

User:
Why are they different from my current salary?

The history-aware retriever reformulates the second question into:

"Why are bonus eligible earnings different from my current salary?"

before retrieving relevant documents.

### Benefits

- Improved retrieval quality for follow-up questions.
- Better user experience.
- Reduced ambiguity in conversational queries.


## Phase 4 – Agentic Orchestration with LangGraph

### Objective

Re-architect the HR Assistant as a LangGraph workflow that intelligently routes user queries based on intent while maintaining a shared conversational state.

The goal of this phase was to move beyond a single retrieval pipeline and introduce decision-making into the application. Depending on the user's query, the system determines whether retrieval from HR documents is required or whether the question can be answered through normal conversation.

---

## Architecture

```text
START
  ↓
Router Node
  ↓
 ┌─────────────┬
 │             │
HR Query   Casual Query
 │             │
 ↓             ↓
HR Node    Casual Node
 │             │
 └──────┬──────┘
        ↓
       END
```

---

## Graph Components

### 1. Router Node

The Router Node is responsible for classifying incoming user queries into one of two categories:

- `hr`
- `casual`

The router uses an LLM-based classification prompt with examples and routing rules.

Examples:

| Query | Route |
|---------|---------|
| What is bonus eligible earnings? | HR |
| How many days of parental leave do I get? | HR |
| Can I bring my dog to the office? | HR |
| Hello | Casual |
| Thank you | Casual |
| What was the first question I asked? | Casual |

Routing Principle:

> If there is any possibility that a query relates to employee policies, benefits, compensation, workplace rules, payroll, leave programs, or HR documentation, the query is routed to the HR branch.

---

### 2. HR Node

The HR Node executes the Conversational RAG pipeline developed in Phase 3.

Components:

- FAISS Vector Store
- OpenAI Embeddings (`text-embedding-3-large`)
- History-Aware Retriever
- Llama 3.1 (Groq)
- Source Attribution

Workflow:

```text
Question
    ↓
History-Aware Retriever
    ↓
FAISS Similarity Search
    ↓
Retrieved Documents
    ↓
LLM
    ↓
Grounded Answer
```

The node returns:

- Answer
- Source Documents

---

### 3. Casual Node

The Casual Node handles:

- Greetings
- Small talk
- Conversation history questions
- General non-HR discussions

Examples:

- Hello
- How are you?
- Thank you
- What was the first question I asked?

No document retrieval is performed.

The node uses conversation history to maintain context across turns.

Workflow:

```text
Chat History
     ↓
LLM
     ↓
Response
```

The node returns:

- Answer
- No Sources

---

## Shared Conversation State

A common graph state is maintained across all nodes.

```python
class GraphState(TypedDict):
    question: str
    answer: str
    sources: list
    chat_history: list
    route: str
```

This ensures that both HR and Casual branches operate on the same conversation history.

Benefits:

- Follow-up questions work correctly.
- Users can switch between HR and casual conversation seamlessly.
- Context is preserved regardless of which branch processes the request.

Example:

```text
User:
What is paid parental leave?

Assistant:
...

User:
What is the time period within which I can use it?
```

The conversation history allows the system to understand that "it" refers to Paid Parental Leave.

---

## Routing Challenges

One of the primary challenges in this phase was handling borderline questions.

Examples:

| Query | Expected Route |
|---------|---------|
| Can I bring my dog to the office? | HR |
| What was the first question I asked? | Casual |
| When can I use parental leave? | HR |
| Tell me a joke | Casual |

To improve routing accuracy:

- Added domain-specific HR examples
- Included leave, payroll, compensation and workplace-policy terminology
- Configured the router to default to HR when uncertain

This reduces the risk of answering HR policy questions without retrieval.

---

## Relevance Score

The assistant displays a relevance score for each retrieved source to provide visibility into the retrieval process.

After the user's question is rewritten into a standalone query by the History-Aware Retriever, embeddings are generated for both the query and the retrieved document chunks. The relevance score is then computed using **cosine similarity**:

```python
relevance_score = cosine_similarity(
    query_embedding,
    chunk_embedding
) * 100
```

Cosine similarity measures the semantic closeness between the query and the retrieved content, where higher values indicate stronger relevance.

| Relevance Score | Interpretation |
|----------------|----------------|
| 90–100% | Very strong semantic match |
| 75–90% | Strong semantic match |
| 60–75% | Moderate semantic match |
| Below 60% | Weak semantic match |

> The relevance score reflects how closely the retrieved document matches the query and should not be interpreted as a measure of answer correctness.
---

## Key Learnings

### LangChain vs LangGraph

Phase 3 used a single retrieval pipeline:

```text
User
 ↓
Retriever
 ↓
LLM
 ↓
Answer
```

Phase 4 introduces branching workflows:

```text
User
 ↓
Router
 ↓
HR? ──► Retrieval Pipeline
 │
 No
 ▼
Conversation Pipeline
```

LangGraph enables:

- Explicit workflow orchestration
- Conditional routing
- Shared state management
- Multi-path execution

---

## Outcome

Successfully implemented an agentic HR assistant that:

- Routes HR-related questions through Conversational RAG.
- Routes non-HR questions through a conversational LLM path.
- Maintains shared conversation history across all interactions.
- Preserves source attribution for HR responses.
- Demonstrates LangGraph-based orchestration and decision making.
