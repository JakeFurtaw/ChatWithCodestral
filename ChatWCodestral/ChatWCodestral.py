from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


def main() -> None:
    llm = Ollama(model="codestral:latest", request_timeout=30.0)
    while (query := input("Enter your coding question(e to exit): ")) != "e":
        question = [ChatMessage(content=query)]
        answer = llm.stream_chat(question)
        print(answer)


if __name__ == "__main__":
    main()
