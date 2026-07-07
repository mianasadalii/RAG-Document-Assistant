# RAG-Document-Assistant

Welcome to the **Smart RAG Document Chatbot**! This application allows you to upload any PDF document and instantly chat with it. It uses an AI technique called RAG (Retrieval-Augmented Generation) to read your document, understand its context, and give you precise answers based *only* on the information inside the file.

---

## ✨ Features

* **Easy PDF Uploads:** Simply drag and drop your PDF file to get started.
* **Instant Processing:** Quickly breaks down your document and builds a searchable knowledge base.
* **Smart Q&A:** Ask questions in plain English, and the AI will fetch the exact answers from your document.
* **No Hallucinations:** The chatbot is strictly instructed to say "I don't have information about this topic" if the answer isn't in the PDF.
* **Blazing Fast:** Powered by Groq's lightning-fast AI models.
* **User-Friendly Interface:** Built with Gradio for a clean and simple chat experience.

---

## 🛠️ How It Works (The Magic Behind the Scenes)

This app uses a process called **RAG (Retrieval-Augmented Generation)**. Here is how it works in simple steps:

1. **Read & Split:** The app reads your uploaded PDF and chops the text into smaller, manageable chunks.
2. **Embed & Store:** It uses Hugging Face to convert these text chunks into numbers (embeddings) and saves them in a highly organized database (FAISS). 
3. **Search & Retrieve:** When you ask a question, the app searches the database to find the text chunks most relevant to your question.
4. **Generate Answer:** It hands your question and the relevant text chunks to the Groq AI model, which reads the context and types out a helpful, accurate answer.

---

## 💻 Tech Stack

* **Frontend:** [Gradio](https://www.gradio.app/) (For the web interface)
* **Orchestration:** [LangChain](https://www.langchain.com/) (To tie all the AI parts together)
* **LLM (Large Language Model):** [Groq](https://groq.com/) (Using `llama-3.1-8b-instant`)
* **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Document Loader:** PyPDF (For reading PDF files)

---
