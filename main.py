import os
import sqlite3
from typing import Any
import pandas as pd
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# from langchain.chains import SQLDatabaseChain
from langchain_experimental.sql import SQLDatabaseChain
# from langchain_community.llms import OpenAI, SQLDatabase
from langchain_huggingface import HuggingFacePipeline
# from langchain.utilities.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain.schema.cache import BaseCache
from langchain.callbacks.base import Callbacks

from utils import setup_logger

# CONSTANTS and ENVIRONMENT VARIABLES
load_dotenv()
MODEL_NAME=os.getenv("MODEL_NAME")



logger = setup_logger("main", "main.log", logging.INFO)


# Mock a simple BaseCache implementation
class SimpleCache(BaseCache):
    def lookup(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass
# Ensure SQLDatabaseChain is fully defined
SQLDatabaseChain.model_rebuild()

def create_llm():
    """
    Create a simple Hugging Face LLM pipeline.
    Replace 'gpt2' with a more suitable finetuned or instruct model if available.
    """
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create a text-generation pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1,
    )

    # Wrap it in a LangChain LLM interface
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


def create_sqlite_db_from_csv(db_path: str, csv_path: str, table_name: str):
    """
    Create a SQLite database from a CSV file if the database does not already exist.
    """
    if not os.path.exists(db_path):
        logger.info(f"Database not found. Creating SQLite database from {csv_path}...")
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        conn.close()
        logger.info(f"Database created and table '{table_name}' populated with data.")
    else:
        logger.info(f"Database found at {db_path}. Skipping creation.")


def main():
    # File paths
    db_path = "data.db"
    csv_path = "data.csv"
    table_name = "transactions"

    # Create the SQLite database from the CSV file if it doesn't exist
    create_sqlite_db_from_csv(db_path, csv_path, table_name)

    # Create an LLM
    llm = create_llm()

    # Connect to the SQLite DB
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # Create a chain that can query this DB
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    # Ask a question in natural language
    questions = [
        "How many rows are in the 'transactions' table?"
    ]
    for question in questions:
        try:
            logger.info("Question:", question)
            logger.info("Generating SQL query...")
            response = db_chain.invoke({"question": question})
            logger.info("Generated SQL Query:", response["sql_query"])
            logger.info("Answer:", response["answer"])
        except Exception as e:
            logger.info("Error:", e)
            raise e


if __name__ == "__main__":
    main()



