from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer, ResponseMode

prompt = PromptTemplate(
    "You are an expert software engineer, coding genius, you can answer any question coding related."
    "Your job is to help the user by giving the user the most accurate up to date information about anything\n"
    "coding related they might ask."
    "Make sure if you dont understand what the user is asking to ask clarifying question, dont answer a question\n"
    "you dont know. If you need more information to fully answer a question, just ask."
)


def main() -> None:
    # retriever = index.as_retreiver()
    synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        verbose=True,
        streaming=True
    )
    query_engine = RAGQueryEngine(
        retriever=retriever, response_synthesizer=synthesizer
    )
    llm = Ollama(model="codestral:latest", request_timeout=30.0)
    while (query := input("Enter your coding question(e to exit): ")) != "e":
        response = query_engine.query(query)
        print(str(response))

def milvus:

class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj


if __name__ == "__main__":
    main()
