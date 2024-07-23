from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate

prompt = PromptTemplate(
    "You are an expert software engineer, coding genius, you can answer any question coding related."
    "Your job is to help the user by giving the user the most accurate up to date information about anything\n"
    "coding related they might ask."
)


def main() -> None:
    llm = Ollama(model="codestral:latest", request_timeout=30.0)
    while (query := input("Enter your coding question(e to exit): ")) != "e":
        question = [ChatMessage(content=query)]
        answer = llm.complete(
            prompt.format(query=query)
        )
        print(answer)


if __name__ == "__main__":
    main()
