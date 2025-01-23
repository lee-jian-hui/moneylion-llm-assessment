import logging
import os
from typing import Any, List
from llama_cpp import Llama

from langchain_experimental.sql import SQLDatabaseChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain.schema.cache import BaseCache
from langchain.callbacks.base import Callbacks
from langchain.sql_database import SQLDatabase
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM

from utils import setup_logger

MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Update this to the actual path of your model file

DATABASE_URL = "sqlite:///data.db"  # Replace with your actual database URL
MAX_TOKENS = 200
TEMPERATURE = 0.4
CONTEXT_WINDOW_SIZE=4096


logger = setup_logger(__name__, "test.log", level=logging.WARNING)

# Custom LLM Wrapper for llama_cpp
class CustomLlamaLLM(LLM):
    def __init__(self, model: Llama):
        super().__init__()
        self._model = model

    @property
    def _llm_type(self) -> str:
        return "llama_cpp"

    def _call(self, prompt: str, stop: list = None, **kwargs: Any) -> str:
        response = self._model(
            prompt, stop=stop, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
        )
        return response["choices"][0]["text"].strip()


# Mock BaseCache implementation
class SimpleCache(BaseCache):
    def lookup(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


# Ensure SQLDatabaseChain is fully defined
SQLDatabaseChain.model_rebuild()


def load_model(model_path) -> Llama:
    """
    Load the Mistral-7B-GGUF model using llama-cpp-python.

    Args:
        model_path (str): Path to the GGUF model file.

    Returns:
        Llama: Loaded model instance.
    """
    try:
        model = Llama(model_path=model_path, n_ctx=CONTEXT_WINDOW_SIZE)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_database_connection(database_url: str = DATABASE_URL) -> SQLDatabase: 
    """
    Load the database connection using LangChain's SQLDatabase module.

    Returns:
        SQLDatabase: A connection to the SQL database.
    """
    try:
        db = SQLDatabase.from_uri(database_url)
        print("Database connected successfully.")
        return db
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def create_banking_assistant(database: SQLDatabase, llm: CustomLlamaLLM) -> SQLDatabaseChain:
    try:
        db_chain = SQLDatabaseChain.from_llm(llm=llm, db=database, verbose=True)

        # 1. To get the raw template string (if applicable)
        try:
            prompt = db_chain.llm_chain.prompt
            print("Raw template string:", prompt.template)
        except AttributeError:
            print("This prompt does not have a `template` attribute.")

        print("Banking assistant created successfully.")
        return db_chain
    except Exception as e:
        print(f"Error creating banking assistant: {e}")
        return None


def chat_loop(banking_assistant: SQLDatabaseChain, simulated: bool = False, questions: List[str] = None):
    """
    Chat loop for interacting with the banking assistant, maintaining context.

    Args:
        banking_assistant (SQLDatabaseChain): The banking assistant instance.
        simulated (bool): If True, uses a predefined list of questions.
        questions (List[str]): List of questions for the simulated mode.
    """
    print("\nWelcome to the Banking Assistant!")
    print("Type your natural language request below, or type 'exit' to quit.")

    # Initialize conversation history
    conversation_history = ""

    if simulated and questions:
        for question in questions:
            print(f"\nYour Query: {question}")
            try:
                conversation_history += f"User: {question}\n"
                response = banking_assistant.run(conversation_history)
                conversation_history += f"Assistant: {response}\n"
                print("\nQuery Result:")
                print(response)
            except Exception as e:
                print(f"Error processing query: {e}")
    else:
        while True:
            user_input = input("\nYour Query: ").strip()

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            try:
                conversation_history += f"User: {user_input}\n"
                response = banking_assistant.run(conversation_history)
                conversation_history += f"Assistant: {response}\n"
                print("\nQuery Result:")
                print(response)
            except Exception as e:
                print(f"Error processing query: {e}")


def test_database_context(sql_llm_chain, database):
    """
    Test if the database context is loaded properly and log the formatted prompt.

    Args:
        sql_llm_chain (SQLDatabaseChain): The SQL LLM chain.
        database (SQLDatabase): The database connection object.
    """
    try:
        # Extract the prompt template from the chain
        prompt = sql_llm_chain.llm_chain.prompt
        logger.info("Raw SQL LLM chain template:\n%s", prompt.template)

        # Example input to test the formatted prompt
        example_input = {
            "input": "Give me all transactions",
            "table_info": database.table_info,
            "top_k": 5,  # Adjust as necessary for your configuration
        }
        logger.info("Example input for prompt:\n%s", example_input)

        # Format the prompt with example input and log it
        formatted_prompt = prompt.format(**example_input)
        logger.info("Formatted prompt:\n%s", formatted_prompt)

    except AttributeError:
        logger.error("This prompt does not have a `template` attribute.")
    except Exception as e:
        logger.error("Error while testing database context: %s", e)


def main():
    # Load the model
    model = load_model(MODEL_PATH)
    if not model:
        logger.error("Model failed to load.")
        return

    # Load the database connection
    database = load_database_connection()
    if not database:
        logger.error("Database connection failed.")
        return

    llama_llm = CustomLlamaLLM(model)

    # Create the banking assistant
    sql_llm_chain = create_banking_assistant(database, llama_llm)
    if not sql_llm_chain:
        logger.error("Failed to create SQL LLM chain.")
        return

    # Test the database context
    # test_database_context(sql_llm_chain, database)

    # Chat loop 
    questions = [
        "How many rows are in the 'transactions' table?",
        "How many transactions are in there again? Can you filter for transactions with merchant=1INFINITE"
    ]
    chat_loop(sql_llm_chain, simulated=True, questions=questions)
    # chat_loop(sql_llm_chain, simulated=False)


if __name__ == "__main__":
    main()
