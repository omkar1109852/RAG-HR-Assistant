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
