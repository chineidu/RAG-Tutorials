# RAG Chat Application 💬

A Streamlit-based web interface for document-based question answering using Retrieval-Augmented Generation (RAG) with advanced query fusion techniques.

## 📖 Overview

This frontend application provides an interactive chat interface where users can upload PDF documents and ask questions about their content. It implements RAG Fusion, an advanced retrieval technique that generates multiple search queries and combines results using Reciprocal Rank Fusion (RRF) for improved answer quality.

## ✨ Key Features

- **Interactive Chat Interface**: Streamlit-powered conversational UI
- **PDF Document Processing**: Upload and process PDF files for question answering
- **RAG Fusion Implementation**: Advanced retrieval using multiple generated queries
- **Reciprocal Rank Fusion**: Sophisticated result ranking and combination
- **Real-time Streaming**: Live response generation with streaming output
- **Model Flexibility**: Support for both local (Ollama) and remote (OpenRouter) models
- **Persistent Collections**: Vector store collections with custom naming
- **Session Memory**: Chat history maintained during the session

## 🛠️ Technologies Used

### UI Framework

- **[Streamlit](https://streamlit.io/)** - Interactive web application framework for the UI

### AI/ML Libraries

- **[LangChain](https://langchain.com/)** - Core framework for LLM applications

### Vector Database & Embeddings

- **[Qdrant](https://qdrant.tech/)** - Vector similarity search engine
- **[Ollama Embeddings](https://ollama.ai/)** - Local embedding models (mxbai-embed-large)

### Document Processing

- **PyPDFLoader** - PDF document loading and parsing
- **RecursiveCharacterTextSplitter** - Text chunking with tiktoken encoding

### Language Models

- **Local Models (via Ollama)**:
  - Mistral 7B Instruct v0.3 (quantized)
  - Llama 3.1/3.2 variants
  - Qwen models
- **Remote Models (via OpenRouter)**:
  - Google Gemini 2.0 Flash
  - Meta Llama models
  - Mistral models
  - Various other open-source models

### Configuration & Utilities

- **[Pydantic Settings](https://docs.pydantic.dev/)** - Configuration management
- **[Python Dotenv](https://pypi.org/project/python-dotenv/)** - Environment variable loading

## 🏗️ Architecture

### RAG Fusion Pipeline

```
User Question → Query Generation → Multi-Query Retrieval → RRF Combination → Answer Generation
```

1. **Query Generation**: Original question is expanded into 3 diverse search queries
2. **Multi-Query Retrieval**: Each query searches the vector store independently
3. **Reciprocal Rank Fusion**: Results are combined and reranked using RRF algorithm
4. **Answer Generation**: Final context is used to generate a comprehensive answer

### Key Components

- **Document Processing**: PDF upload → text extraction → chunking → embedding → vector storage
- **Query Fusion**: Single query → multiple semantically diverse queries
- **Retrieval**: Vector similarity search across multiple query variations
- **Ranking Fusion**: Reciprocal Rank Fusion for result combination
- **Response Generation**: Context-aware answer generation with streaming

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Streamlit
- Qdrant vector database (running locally or via Docker)
- Ollama (for local models) or API keys for remote models

### Installation

1. **Navigate to the frontend directory**:

   ```bash
   cd frontend
   ```

2. **Install dependencies** (from project root):

   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Start Qdrant** (if not already running):

   ```bash
   # From project root
   docker-compose --profile retriever up -d
   ```

4. **Configure environment variables**:

   ```bash
   # Set up your .env file with required API keys
   OPENROUTER_API_KEY=your_openrouter_key
   OLLAMA_API_KEY=your_ollama_key  # Optional for local
   ```

### Running the Application

```bash
# From project root
uv run streamlit run frontend/main.py

# Or if using pip
streamlit run frontend/main.py
```

The application will be available at `http://localhost:8501`

## 💻 Usage

### 1. Upload Document

- Use the sidebar file uploader to select a PDF document
- Enter a custom collection name (or use "demo" default)
- Click "Create Collection" to process the document

### 2. Chat Interface

- Once the document is processed, use the chat input at the bottom
- Ask questions about the uploaded document
- Receive AI-generated answers with relevant context
- Chat history is maintained during the session

### 3. Advanced Features

- **RAG Fusion**: Automatically generates multiple search queries for better retrieval
- **Streaming Responses**: See answers being generated in real-time
- **Source Attribution**: Answers are based on specific document sections
- **Model Selection**: Configure local vs. remote models in settings

## ⚙️ Configuration

### Model Configuration (`model_config.py`)

**Local Models (Ollama)**:

- `mistral:7b-instruct-v0.3-q4_0` (default)
- `llama3.1:8b`, `llama3.2:3b`
- `qwen3-1.7b`, `qwen3:4b`

**Remote Models (OpenRouter)**:

- `google/gemini-2.0-flash-001` (default)
- `meta-llama/llama-3.3-70b-instruct`
- `mistralai/mistral-7b-instruct-v0.3`

**Embedding Model**:

- `mxbai-embed-large:latest` (via Ollama)

### Settings Configuration

Key settings in `settings.py`:

- `QDRANT_URL`: Vector database connection
- `OLLAMA_URL`: Local Ollama server URL
- `OPENROUTER_URL`: Remote model API endpoint
- API keys for various services

### RAG Parameters

- **Chunk Size**: 500 tokens with 100 token overlap
- **Retrieval Count**: Top 3 most relevant chunks per query
- **Query Generation**: 3 diverse queries per user question
- **RRF Parameter**: k=60 for rank fusion scoring

## 🔧 Customization

### Modifying the RAG Pipeline

The application uses a modular design allowing easy customization:

- **Query Generation**: Modify the prompt template in `get_rag_fusion_generator()`
- **Chunking Strategy**: Adjust parameters in `get_document_splits()`
- **Retrieval Parameters**: Change search_kwargs in `get_vector_store()`
- **Answer Generation**: Update the RAG prompt in `get_final_rag_pipeline()`

### Adding New Models

To add new models:

1. Add model definitions to `LocalModel` or `RemoteModel` enums
2. Update the model selection logic in `main.py`
3. Configure appropriate API endpoints and keys

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**: Ensure Qdrant is running on the configured URL
2. **Model Not Found**: Verify Ollama models are pulled locally
3. **API Key Issues**: Check environment variable configuration
4. **Memory Issues**: Reduce chunk size or document length for large files

### Performance Tips

- Use quantized models (q4_0) for faster local inference
- Adjust chunk size based on document complexity
- Consider using remote models for better quality vs. local for privacy

## 📄 File Structure

```
frontend/
└── main.py          # Main Streamlit application
```

**Dependencies** (defined in project root):

- Core: `streamlit`, `langchain-*`, `qdrant-client`
- Document Processing: `PyPDFLoader`, `RecursiveCharacterTextSplitter`
- Models: `langchain-ollama`, `langchain-openai`
- Configuration: `pydantic-settings`, `python-dotenv`

## 🎯 Future Enhancements

Potential improvements for the frontend:

1. **Multi-document Support**: Handle multiple PDFs in a single collection
2. **Advanced Filtering**: Filter results by document source or page
3. **Export Features**: Export chat conversations or citations
4. **Visualization**: Display similarity scores and source highlighting
5. **User Management**: Support for multiple users and sessions

---

**Ready to chat with your documents!** 📚✨
