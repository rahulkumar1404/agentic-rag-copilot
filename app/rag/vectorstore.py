import time 
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
    "gemini-embedding-2-preview": 3072,
}

def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("No embedding model specified")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]

    raise ValueError(
        f"Unsupported embedding model: '{name}' for Pinecone. "
        "Add the matching embedding dimension to EMBEDDING_DIMENSIONS."
    )
    
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.google_api_key:
            raise RuntimeError("Google API key is not set")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.google_api_key
        )
    return _embeddings

def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("Pinecone API key is not set")
    if not settings.pinecone_index_name:
        raise RuntimeError("Pinecone index name is not set")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name
    names = [item["name"] for item in pc.list_indexes()]

    if name not in names:
        pc.create_index(
            name=name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    deadline = time.monotonic() + 120

    while True:
        index_info = pc.describe_index(name)

        if index_info.dimension != desired_dimension:
            raise RuntimeError(
                f"Index '{name}' has dimension {index_info.dimension}; "
                f"expected {desired_dimension}. Choose a compatible index."
            )

        if index_info.status["ready"]:
            break

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index '{name}' was not ready within 120 seconds."
            )

        time.sleep(1)

    return pc.Index(name)

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace
        )
    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k":settings.top_k}
    )
    

def add_document(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)