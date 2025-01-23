import os
import sqlite3
from typing import Any
import pandas as pd
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
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
# File paths
DB_PATH = "data.db"
CSV_PATH = "data.csv"
TABLE_NAME = "transactions"



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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)  # Use Seq2Seq model

    # Create a text-generation pipeline
    # hf_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=100,
    #     do_sample=True,
    #     temperature=0.1,
    # )

    hf_pipeline = pipeline(
        "text2text-generation",  # Use text2text-generation for seq2seq models
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

    # Create the SQLite database from the CSV file if it doesn't exist
    create_sqlite_db_from_csv(DB_PATH, CSV_PATH, TABLE_NAME)

    # Create an LLM
    llm = create_llm()

    # Connect to the SQLite DB
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    logger.info(f"db.dialect: {db._engine.dialect.name}")

    # Create a chain that can query this DB
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    prompt = db_chain.llm_chain.prompt

    # 1. To get the raw template string (if applicable)
    try:
        print("Raw template string:", prompt.template)
    except AttributeError:
        print("This prompt does not have a `template` attribute.")

    # Ask a question in natural language
    questions = [
        "How many rows are in the 'transactions' table?"
    ]
    for question in questions:
        try:
            logger.info(f"Question: {question}")
            logger.info("Generating SQL query...")
            response = db_chain.invoke({"query": question})
            # response = db_chain.invoke({"question": question})
            logger.info(f'response: {response}')
            logger.info(f'Generated SQL Query: {response["sql_query"]}')
            logger.info(f'Answer: {response["answer"]}')
        except Exception as e:
            logger.info("Error:", e)
            raise e


def testing():
    # Create the SQLite database from the CSV file if it doesn't exist
    create_sqlite_db_from_csv(DB_PATH, CSV_PATH, TABLE_NAME)

    # Create an LLM
    llm = create_llm()

    # Connect to the SQLite DB
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    # Create a chain that can query this DB
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    # prompt_template = PromptTemplate(
    #     input_variables=["country"],
    #     template=SQLDatabaseChain.prompt
    # )
    # # Fill the prompt with variables
    # filled_prompt = prompt_template.format(country="France")
    # print(filled_prompt)

    from langchain.prompts import PromptTemplate

    # Assuming db_chain.llm_chain.prompt is a BasePromptTemplate
    prompt = db_chain.llm_chain.prompt

    # 1. To get the raw template string (if applicable)
    try:
        print("Raw template string:", prompt.template)
    except AttributeError:
        print("This prompt does not have a `template` attribute.")

    # # 2. To format the prompt with variables
    # # Replace `key=value` pairs with the actual input variables expected by your prompt
    # formatted_prompt = prompt.format(**{"key1": "value1", "key2": "value2"})
    # print("Formatted prompt:", formatted_prompt)


    
    # 4. Ask a question in natural language
    question = "How many rows are in the 'transactions' table?"
    response = db_chain.run(question)
    print("Answer:", response)





if __name__ == "__main__":
    main()
    # testing()
