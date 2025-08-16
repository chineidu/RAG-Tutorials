# uvr -m streamlit run frontend/main.py

import os
import tempfile
from operator import itemgetter
from typing import Any, Generator
from uuid import uuid4

import streamlit as st
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from model_config import LocalModel, RemoteModel
from settings import refresh_settings

settings = refresh_settings()
model_str_remote: str = RemoteModel.GEMINI_2_0_FLASH_001
model_str_local: str = LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0


# Deterministic responses
remote_llm = ChatOpenAI(
    api_key=settings.OPENROUTER_API_KEY.get_secret_value(),  # type: ignore
    base_url=settings.OPENROUTER_URL,  # type: ignore
    temperature=0.0,
    model=model_str_remote,  # type: ignore
)

local_llm = ChatOpenAI(
    api_key=settings.OLLAMA_API_KEY.get_secret_value(),  # type: ignore
    base_url=settings.OLLAMA_URL,  # type: ignore
    temperature=0.0,
    model=model_str_local,  # type: ignore
)

llm = remote_llm


def get_uploaded_filepath(uploaded_file: Any) -> str:
    """Get the file path of the uploaded PDF."""
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
    return tmp_file.name


def remove_uploaded_file(file_path: str) -> None:
    """Remove the uploaded file from the filesystem."""
    if file_path is not None:
        os.unlink(file_path)


def get_document_splits(filepath: str) -> list[Any]:
    """Load a PDF file and split it into smaller chunks."""
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(docs)


def get_vector_store(splits: list[Any], collection_name: str) -> VectorStoreRetriever:
    """Create a vector store retriever from document splits."""
    emb_model = OllamaEmbeddings(
        model=LocalModel.MXBAI_EMBED_LARGE,
    )
    emb = emb_model.embed_documents(["Hello world"])
    emb_size: int = len(emb[0])

    client = QdrantClient(url=settings.QDRANT_URL)
    collection_exists_flag: bool = client.collection_exists(collection_name=collection_name)

    if not collection_exists_flag:
        client = QdrantClient(url=settings.QDRANT_URL)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=emb_size, distance=Distance.COSINE),
        )

        # Vector store
        vector_store: QdrantVectorStore = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=emb_model,
            collection_name=collection_name,
            ids=[str(uuid4()) for _ in range(len(splits))],
        )
    else:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=emb_model,
            collection_name=collection_name,
            url=settings.QDRANT_URL,
        )

    return vector_store.as_retriever(search_kwargs={"k": 3})


def get_rag_fusion_generator(llm: ChatOpenAI) -> RunnableSerializable[dict, Any]:
    """Generate RAG fusion queries."""
    template = """
    <system>
        <role>
        You are a helpful query reconstructor that generates multiple search queries based on a 
        single input query to improve document retrieval.
        </role>

        <instructions>
        Generate exactly 3 distinct search queries that would help find relevant information for the given question.
        
        Original question: {question}
        
        Your search queries should:
        - Be specific and use actual entities/terms from the original question
        - NOT use placeholders. Use the actual entities mentioned
        - Cover different angles or phrasings of the same information need
        - Be suitable for semantic search in documents
        - The generated queries should be on a new line
        </instructions>

        <outputs>
        Output:
        </outputs>
    </system>
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    return prompt_rag_fusion | llm | StrOutputParser() | (lambda x: x.strip().split("\n"))


def reciprocal_rank_fusion(results: list[list], k: int = 60, num_results: int = 4) -> list[tuple[Any, Any]]:
    """
    Apply Reciprocal Rank Fusion (RRF) to combine multiple ranked document lists.
    """
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    sorted_docs = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_docs[:num_results]


def extract_context_and_metadata(
    docs_with_scores: list[tuple[Any, float]],
) -> dict[str, Any]:
    """Extract both context and metadata from RAG fusion output."""
    docs = [doc for doc, _ in docs_with_scores]
    context = "\n\n".join(
        f"(content: {doc.page_content}\nsource: {doc.metadata['source']}\npage_label: {doc.metadata['page_label']})"
        for doc in docs
    )
    result = list({(doc.metadata["source"], doc.metadata["page_label"]) for doc in docs})
    metadata = [{"source": row[0], "page_label": row[1]} for row in result]
    return {"context": context, "metadata": metadata}


def combine_with_question(inputs: dict[str, Any]) -> dict[str, Any]:
    """Combine retrieval results with the original question."""
    question = inputs["question"]
    retrieval_results = inputs["retrieval_results"]
    return {
        "question": question,
        "context": retrieval_results["context"],
        "metadata": retrieval_results["metadata"],
    }


def get_final_rag_pipeline(retrieval_chain: Any, llm_model: Any) -> tuple[RunnableSerializable, RunnableSerializable]:
    """Generate the final RAG pipeline with streaming support."""
    # RAG Prompt
    rag_template: str = """
    <instructions>
    You are a helpful AI assistant. Answer the following question based primarily on the provided context.
    
    <context>{context}</context>
    <question>{question}</question>
    
    If the context doesn't contain relevant information, you may use your general knowledge to provide a 
    helpful response.
    </instructions>

    <guidelines>
    - Prioritize information from the provided context when available
    - Keep responses concise (maximum 3 sentences unless the user requests more detail)
    - Be accurate and relevant to the question asked
    - If you cannot answer based on context or knowledge, respond with "I don't know"
    - Respond naturally to greetings and casual conversation
    - Use a friendly, professional tone
    </guidelines>
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    return (
        {
            "context": retrieval_chain,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | llm_model
        | StrOutputParser()
    )


# Streamlit UI
st.title("Chat With Your Documents")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        tmp_file_path = get_uploaded_filepath(uploaded_file)

        with st.form(key="rag_form", clear_on_submit=True):
            collection_name = st.text_input("Collection Name", value="demo")
            submit_button = st.form_submit_button("Create Collection")

            if submit_button:
                # Process the document
                with st.spinner("Processing document..."):
                    splits = get_document_splits(tmp_file_path)
                    generate_queries = get_rag_fusion_generator(llm=llm)
                    # Create or fetch collection
                    retriever = get_vector_store(splits, collection_name)
                    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
                    st.session_state.rag_chain = get_final_rag_pipeline(retrieval_chain_rag_fusion, llm_model=llm)
                    st.success(
                        f"Collection '{collection_name}' created successfully and processed {len(splits)} document chunks!"
                    )
                    # Remove temporary file
                    remove_uploaded_file(tmp_file_path)

    else:
        st.warning("Please create a collection first!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's on your mind?"):
    if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
        st.warning("Please upload a PDF file first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                def response_generator() -> Generator[str, Any, None]:
                    """Generate response chunks from the RAG chain."""
                    # Stream the LLM response
                    for chunk in st.session_state.rag_chain.stream({"question": prompt}):
                        if chunk:  # Only yield non-empty chunks
                            yield chunk

                response = st.write_stream(response_generator())

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
