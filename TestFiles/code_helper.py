from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are an expert software engineer, coding genius, you can answer any question coding related."
            "Your job is to help the user by giving the user the most accurate up to date information about anything\n"
            "coding related they might ask."
        ),
    ),
    ChatMessage(role=MessageRole.USER, content=qa_prompt_str),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are an expert software engineer, coding genius, you can answer any question coding related."
            "Your job is to help the user by giving the user the most accurate up to date information about anything\n"
            "coding related they might ask."
        ),
    ),
    ChatMessage(role=MessageRole.USER, content=refine_prompt_str),
]
refine_template = ChatPromptTemplate(chat_refine_msgs)


def main() -> None:
    llm = Ollama(model="codestral:latest", request_timeout=30.0)
    while (query := input("Enter your coding question(e to exit): ")) != "e":
        question = [ChatMessage(content=query)]
        answer = llm.stream_chat(question)
        print(answer)


if __name__ == "__main__":
    main()
