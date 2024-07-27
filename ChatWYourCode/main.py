from utils import load_embedding_model, initialize_github_client, set_device
from llama_index.readers.github import GithubRepositoryReader
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
import os


def main() -> None:
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    # loading embedding model and dotenv files
    lc_embedding_model = load_embedding_model()
    embed_model = LangchainEmbedding(lc_embedding_model)
    load_dotenv()

    # -------------------------------
    # -----Swap Repo Info Here-------
    # -------------------------------
    owner = "JakeFurtaw"
    repo = "Oceans"
    branch = "main"
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = initialize_github_client(github_token)
    loader = GithubRepositoryReader(
        github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=(
            [".js", "jsx", ".css"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        use_parser=False,
        verbose=False,
        concurrent_requests=20
    )
    docs = loader.load_data(branch=branch)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    # Setting Ollama Model Here
    llm = Ollama(model="codestral:latest", request_timeout=30.0)
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
    # -------------------------------
    # ---Question About Repo Here----
    # -------------------------------
    response = query_engine.query('What is this repository about?')
    print(response)


if __name__ == "__main__":
    main()
