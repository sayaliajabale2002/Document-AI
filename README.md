# Document-AI

This project, named **Document-AI**, is a Retrieval-Augmented Generation (RAG) system built with Streamlit, LangGraph, and LangChain that uses Google's `gemini-2.5-flash-lite` model and `gemini-embedding-001` to answer questions directly from uploaded PDF documents. It works by chunking the PDF using a recursive text splitter, creating vector embeddings stored via Chroma DB, and executing a specialized LangGraph workflow containing a retrieval node and an LLM generation node to deliver precise answers based strictly on the document context. 

To get started:
1) Navigate into the project folder
2) Set up a Python virtual environment
3) Install the dependencies with pip install -r requirements.txt
4) Create a .env file in the root directory containing your GOOGLE_API_KEY=your-key-here
5) Launch the web application locally by running streamlit run app.py 
