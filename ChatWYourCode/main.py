from utils import (
    initialize_github_client,
    create_qa_prompt_template, initialize_github_loader,
    setup_index_and_query_engine, load_environment_and_models,
    setup_index_and_chat_engine
)
from llama_index.readers.github import GithubRepositoryReader
import os


def main() -> None:
    embed_model, llm = load_environment_and_models()

    owner = "JakeFurtaw"
    repo = "hotlines_mobile"
    branch = "master"
    filter_file_extensions = (
        [".dart"],
        GithubRepositoryReader.FilterType.INCLUDE,
    )
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = initialize_github_client(github_token)

    loader = initialize_github_loader(github_client, owner, repo, filter_file_extensions)
    docs = loader.load_data(branch=branch)

    # Query Engine
    query_engine = setup_index_and_query_engine(docs, embed_model, llm)
    qa_prompt_tmpl = create_qa_prompt_template()
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    # Chat Engine, Added Memory
    chat_engine = setup_index_and_chat_engine(docs, embed_model, llm)
    qa_prompt_tmpl = create_qa_prompt_template()
    chat_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    while True:
        user_query = input("Enter your question about the repository (or e to exit): ")
        if user_query.lower() == 'e':
            print("Exiting the program. Goodbye!")
            break

        response = chat_engine.query(user_query)
        print("Response:", response)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
