from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core import __version__ as lanchain_core_version
from langchain import __version__ as lg_version
from langchain_anthropic import ChatAnthropic


def main():
    print("Hello from rag-freecodecamp!")
    print(f"Langchain version: {lg_version}")
    print(f"Langchain-core version: {lanchain_core_version}")

    # Test the LLM with a simple prompt
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    response = llm.invoke("What is the capital of France?")
    print(f"LLM response: {response}")


if __name__ == "__main__":
    main()
