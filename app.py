import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Global variables to persist the vector database state across chat turns
vector_store = None
retrieval_chain = None

def process_document(file_obj):
    """
    Loads the uploaded PDF, splits it into chunks, creates embeddings,
    and initializes the vector store and retrieval chain.
    """
    global vector_store, retrieval_chain
    
    if file_obj is None:
        return "❌ No file uploaded. Please upload a valid PDF document."
    
    # Check for API key presence to give user a helpful warning if missing
    if not os.environ.get("GROQ_API_KEY"):
        return "❌ GROQ_API_KEY is missing. Please add it to your Hugging Face Space Secrets."
    
    try:
        # 1. Load the document (Supports PDF out-of-the-box via PyPDFLoader)
        loader = PyPDFLoader(file_obj.name)
        documents = loader.load()
        
        # 2. Split the document into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # 3. Download a highly efficient, lightweight embedding model from HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 4. Build and store chunks into the FAISS Vector Database
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 5. Initialize the Groq LLM using active 'llama-3.1-8b-instant' model
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
        
        # 6. Define strict system prompt boundaries for context grounding
        system_prompt = (
            "You are a helpful assistant strictly answering questions based on the provided document context.\n"
            "Answer the user's question using ONLY the context provided below. "
            "If the answer cannot be found in the context, exactly reply: "
            "'I don't have information about this topic. Please ask questions specifically related to the document.'\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 7. Assemble the final QA retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return f"✅ Document processed successfully! Created {len(chunks)} text chunks and updated vector database."
        
    except Exception as e:
        return f"❌ Error processing document: {str(e)}"

def predict(message, history):
    """
    Handles user inquiries via the chat interface using the compiled retrieval chain.
    """
    global retrieval_chain
    
    if retrieval_chain is None:
        return "Please upload and process a document before starting the conversation."
    
    try:
        # Run the RAG pipeline
        response = retrieval_chain.invoke({"input": message})
        return response["answer"]
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Smart RAG Document Chatbot")
    gr.Markdown("Upload any PDF document, process its contents for free, and interrogate it using Groq LLM orchestration.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Document (PDF)", file_types=[".pdf"])
            process_btn = gr.Button("Build Knowledge Base", variant="primary")
            status_output = gr.Textbox(label="Backend Processing Status", value="Awaiting document...", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=predict
            )

    # Wire up button interaction
    process_btn.click(
        fn=process_document,
        inputs=[file_input],
        outputs=[status_output]
    )

# Launch the app for Hugging Face Spaces
demo.launch()
