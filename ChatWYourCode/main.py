from utils import (
    initialize_github_client,
    create_qa_prompt_template, initialize_github_loader,
    setup_index_and_query_engine, load_environment_and_models
)
import os


def main() -> None:
    embed_model, llm = load_environment_and_models()

    owner = "JakeFurtaw"
    repo = "Oceans"
    branch = "main"
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = initialize_github_client(github_token)

    loader = initialize_github_loader(github_client, owner, repo, branch)
    docs = loader.load_data(branch=branch)

    query_engine = setup_index_and_query_engine(docs, embed_model, llm)
    qa_prompt_tmpl = create_qa_prompt_template()
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

    response = query_engine.query('What does the component NewPost.jsx do?')
    print(response)


if __name__ == "__main__":
    main()
