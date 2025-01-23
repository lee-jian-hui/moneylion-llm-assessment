import os
from typing import Any
from llama_cpp import Llama

from langchain_experimental.sql import SQLDatabaseChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain.schema.cache import BaseCache
from langchain.callbacks.base import Callbacks
from langchain.sql_database import SQLDatabase
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM

MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Update this to the actual path of your model file

DATABASE_URL = "sqlite:///banking.db"  # Replace with your actual database URL

# Custom LLM Wrapper for llama_cpp
class CustomLlamaLLM(LLM):
    def __init__(self, model: Llama):
        super().__init__()
        self.model = model

    @property
    def _llm_type(self) -> str:
        return "llama_cpp"

    def _call(self, prompt: str, stop: list = None) -> str:
        response = self.model(prompt, stop=stop, max_tokens=200, temperature=0.5)
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
        model = Llama(model_path=model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_database_connection():
    """
    Load the database connection using LangChain's SQLDatabase module.

    Returns:
        SQLDatabase: A connection to the SQL database.
    """
    try:
        # Example database connection string (use your actual database details)
        database_url = "sqlite:///banking.db"  # Replace with your database URL
        db = SQLDatabase.from_uri(database_url)
        print("Database connected successfully.")
        return db
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def create_banking_assistant(database, model):
    """
    Create a custom SQLDatabaseChain using the Mistral model.

    Args:
        database (SQLDatabase): A connection to the SQL database.
        model (Llama): The loaded Mistral-7B-GGUF model.

    Returns:
        SQLDatabaseChain: A chain for querying the database using natural language.
    """
    try:
        db_chain = SQLDatabaseChain.from_llm(llm=model, db=database, verbose=True)
        # db_chain = SQLDatabaseChain(llm=model, database=database, verbose=True)
        print("Banking assistant created successfully.")
        return db_chain
    except Exception as e:
        print(f"Error creating banking assistant: {e}")
        return None

def main():
    # Path to the GGUF model file

    # Load the model
    model = load_model(MODEL_PATH)
    if not model:
        return

    # Load the database connection
    database = load_database_connection()
    if not database:
        return

    # Create the banking assistant
    banking_assistant = create_banking_assistant(database, model)
    if not banking_assistant:
        return

    print("\nWelcome to the Banking Assistant!")
    print("Type your natural language request below, or type 'exit' to quit.")

    while True:
        # Get user input
        user_input = input("\nYour Query: ").strip()
        
        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # Generate response from the assistant
            response = banking_assistant.run(user_input)
            print("\nQuery Result:")
            print(response)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
