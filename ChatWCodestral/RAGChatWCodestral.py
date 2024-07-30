from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ChatWYourCode.utils import setup_index_and_chat_engine


def main() -> None:
    # Setting up LLM and Embedding models
    llm = Ollama(model="codestral:latest",
                 request_timeout=30.0,
                 device="cuda:0")
    embed_model = HuggingFaceEmbedding(
        model_name="dunzhang/stella_en_1.5B_v5",
        device="cuda:1"
    )
    # Reading and creating doc dictionary
    reader = SimpleDirectoryReader(input_dir="INSERT DIRECTORY HERE", recursive=True).load_data()
    all_docs = []
    for docs in reader:
        for doc in docs:
            all_docs.append(doc)
    # Setting up chat engine
    chat_engine = setup_index_and_chat_engine(docs=all_docs, llm=llm, embed_model=embed_model)
    # Looping chat until user is done chatting
    while (query := input("Enter your coding question(e to exit): ")) != "e":
        response = chat_engine.chat(query)
        print(str(response))


if __name__ == "__main__":
    main()
