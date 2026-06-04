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
