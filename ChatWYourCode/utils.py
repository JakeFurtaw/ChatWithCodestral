from llama_index.readers.github import GithubClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import torch


def initialize_github_client(github_token):
    return GithubClient(github_token)


def load_embedding_model(
        model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"
) -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model


def set_device(gpu: int = None) -> str:
    if torch.cuda.is_available() and gpu is not None:
        device = f"cuda:{gpu}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device
