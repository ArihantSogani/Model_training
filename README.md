# Advanced Transformer PDF Q&A System

This project is an advanced Question Answering (Q&A) system that allows you to upload a PDF document and ask natural language questions about its content. The system leverages state-of-the-art transformer models for document chunking, semantic search, and answer generation, providing concise and human-readable answers.

---

## Features

- **PDF Parsing:** Extracts and cleans text from PDF files.
- **Intelligent Chunking:** Splits the document into overlapping, semantically meaningful chunks.
- **Semantic Embeddings:** Uses Sentence Transformers for high-quality chunk embeddings.
- **Context Retrieval:** Finds the most relevant chunks for a given question using vector search and cross-encoder re-ranking.
- **Answer Generation:** Synthesizes answers using a T5-based generative model, with fallback to extractive QA if needed.
- **Interactive Q&A:** Command-line interface for asking questions about the uploaded PDF.

---

## Models Used

- **Sentence Embeddings:** `all-mpnet-base-v2` (Sentence Transformers)
- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generative QA:** `google/flan-t5-base`
- **Extractive QA:** `deepset/roberta-base-squad2`

---

## Setup

1. **Install Dependencies**

   In your Colab or local environment, run:
   ```python
   !pip install torch transformers sentence-transformers chromadb PyPDF2 nltk
   ```

2. **Download NLTK Data**

   The script will automatically download required NLTK resources (`punkt`, `stopwords`).

3. **Upload Your PDF**

   When prompted, upload your PDF file (e.g., `CompaniesAct2013.pdf`).

---

## Usage

1. **Run the Script**

   Execute the Python script. It will:
   - Initialize all models.
   - Ask you to upload a PDF.
   - Train on the PDF (extract, clean, chunk, and embed).
   - Start an interactive Q&A session.

2. **Ask Questions**

   Type your questions about the document in the prompt. For example:
   ```
   Your question: What is the Companies Act 2013?
   ```

   The system will return a concise, synthesized answer.

3. **Exit**

   Type `quit`, `exit`, or `q` to end the session.

---

## Notes

- The first time you run the script, model downloads may take a few minutes.
- For best results, use well-formatted, text-based PDFs (not scanned images).
- The system is designed for English-language documents and questions.

---
