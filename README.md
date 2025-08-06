# Smart PDF Chatbot

This project is a Streamlit-based AI chatbot that allows you to upload a PDF document and ask questions about its contents. It's built using LangChain, Ollama, ChromaDB, and HuggingFace embeddings for a fast and intelligent retrieval-augmented chatbot experience.

## Features

- Chat with your PDF documents
- Supports conversation memory (context-aware)
- Customizable assistant personality via system prompt
- Built-in vector database with Chroma
- Uses `gemma2:2b` model via Ollama
- Lightweight and fast using `all-MiniLM-L6-v2` embeddings
- Fully local and private (no cloud APIs needed)

## How It Works

1. Upload a PDF  
   → Text is extracted and split into chunks.

2. Create Embeddings  
   → Chunks are embedded using `all-MiniLM-L6-v2`.

3. Store in Vector DB  
   → Chroma DB is used to store and query vectorized text.

4. Chat Interface  
   → User inputs are answered based on relevant PDF chunks.

5. Conversational Memory  
   → Maintains multi-turn chat history with context.

## Tech Stack

| Component         | Technology                    |
|------------------|-------------------------------|
| Web Interface     | Streamlit                     |
| LLM               | Ollama with `gemma2:2b`       |
| Embeddings        | HuggingFace - `all-MiniLM-L6-v2` |
| Vector DB         | ChromaDB                      |
| PDF Extraction    | PyPDF2                        |
| LangChain         | Retrieval + Memory Chains     |
