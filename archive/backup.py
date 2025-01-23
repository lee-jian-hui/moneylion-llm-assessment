import os
import sqlite3
from typing import Any
import pandas as pd
import logging
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dotenv import load_dotenv

# LangChain Imports
from langchain_experimental.sql import SQLDatabaseChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain.schema.cache import BaseCache
from langchain.callbacks.base import Callbacks

from utils import log_resource_usage, setup_logger

# Load environment variables
load_dotenv()
# MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/CodeLlama-7b-hf")  # Default to CodeLlama-7b
DB_PATH = "data.db"
CSV_PATH = "data.csv"
TABLE_NAME = "transactions"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Add your token to the environment
MODEL_NAME_TO_USE = input("what model do you want to use?") or "mistral-7b"
logging.info(f"MODEL_NAME_TO_USE: {MODEL_NAME_TO_USE}")
MODELS = {
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.1", 
        "type": "causal"
    },
    "mistral-7b-gguf": {
        "model_name": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", 
        "type": "causal"
    },
    "llama-7b": {
        "model_name": "meta-llama/Llama-2-7b-hf", 
        "type": "causal"
    },
    "code-llama-7b": {
        "model_name": "meta-llama/CodeLlama-7b-hf",
        "type": "causal"
    }
}


# Logger
logger = setup_logger("main", "main.log", logging.INFO)


# Mock BaseCache implementation
class SimpleCache(BaseCache):
    def lookup(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


# Ensure SQLDatabaseChain is fully defined
SQLDatabaseChain.model_rebuild()


def resolve_model_and_create_pipeline(model_name: str, device: str = "cpu"):
    """
    Automatically resolve the model type and create the appropriate Hugging Face pipeline.
    Args:
        model_name (str): The name of the model to load.
        device (str): The device to load the model on ("cuda:0" for GPU or "cpu").
    """
    logger.info(f"Resolving model type for: {model_name}")
    config = AutoConfig.from_pretrained(model_name, token=os.getenv("HF_AUTH_TOKEN"))
    logger.info(f"config: {config}")


    # Determine the model architecture
    if config.architectures and any("CausalLM" in arch for arch in config.architectures):
        model_class = AutoModelForCausalLM
        task = "text-generation"
        logger.info(f"Model {model_name} resolved as Causal Language Model (CausalLM).")
    elif config.architectures and any("Seq2SeqLM" in arch for arch in config.architectures):
        from transformers import AutoModelForSeq2SeqLM
        model_class = AutoModelForSeq2SeqLM
        task = "text2text-generation"
        logger.info(f"Model {model_name} resolved as Seq2Seq Language Model.")
    # special cases
    elif MODEL_NAME_TO_USE == "mistral-7b-gguf":
        model_class = AutoModelForCausalLM
        task = "text-generation"
        logger.info(f"Model {model_name} resolved as Causal Language Model (CausalLM).")
    else:
        raise ValueError(f"Unsupported model type for {model_name}. Please provide a valid model.")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_AUTH_TOKEN"))
    log_resource_usage(logger)
    model = model_class.from_pretrained(
        model_name,
        # device_map="auto",  # Use "auto" for CUDA, None for CPU
        device_map=None if device == "cpu" else "auto",  # Use "auto" for CUDA, None for CPU
        torch_dtype="auto" if device != "cpu" else None,  # Mixed precision for CUDA only
        low_cpu_mem_usage=True,
        # gpu_layers=0,
        token=os.getenv("HF_AUTH_TOKEN"),
    )
    log_resource_usage(logger)

    # Create the pipeline with the specified device
    hf_pipeline = pipeline(
        task,
        model=model,
        # tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.1,
        device=0 if device.startswith("cuda") else -1,  # CUDA: device=0, CPU: device=-1
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)

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


def prompt_template():
    """
    Returns a prompt template for SQL generation.
    """
    return """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, 
    then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. 
    You can order the results to return the most informative data in the database. Never query for all columns from a table. 
    You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. 
    Also, pay attention to which column is in which table.
    Pay attention to use date('now') function to get the current date, if the question involves "today"."""


def chat_loop(llm, db_chain, simulated: bool = False):
    """
    Looping chat system with streaming and non-streaming options.
    """
    if not simulated:
        while True:
            question = input("\nEnter your question (or type 'exit' to quit): ").strip()
            if question.lower() == "exit":
                print("Goodbye!")
                break
            try:
                print("\nGenerating SQL query...")
                response = db_chain.run(question)
                print(f"Answer: {response}")
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print("An error occurred. Please try again.")
    else:
        questions = [
            "How many rows are in the 'transactions' table?",
            "exit"
        ]
        for question in questions:
            if question.lower() == "exit":
                print("Goodbye!")
                break
            try:
                print("\nGenerating SQL query...")
                response = db_chain.run(question)
                print(f"Answer: {response}")
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print("An error occurred. Please try again.")


def main():
    # Create the SQLite database from the CSV file if it doesn't exist
    create_sqlite_db_from_csv(DB_PATH, CSV_PATH, TABLE_NAME)

    # Resolve and create the LLM pipeline for CodeLlama
    llm = resolve_model_and_create_pipeline(MODELS.get(MODEL_NAME_TO_USE).get("model_name"), device="cpu")

    # Connect to the SQLite DB
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    logger.info(f"Database dialect: {db._engine.dialect.name}")

    # Create a chain that can query this DB
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    # Assuming db_chain.llm_chain.prompt is a BasePromptTemplate
    prompt = db_chain.llm_chain.prompt

    # 1. To get the raw template string (if applicable)
    try:
        print("Raw template string:", prompt.template)
    except AttributeError:
        print("This prompt does not have a `template` attribute.")

    # Start the chat loop for Llama Code model
    chat_loop(llm, db_chain, simulated=True)




def testing():
    # Create the SQLite database from the CSV file if it doesn't exist
    create_sqlite_db_from_csv(DB_PATH, CSV_PATH, TABLE_NAME)

    # Create an LLM
    llm = resolve_model_and_create_pipeline(MODELS.get(MODEL_NAME_TO_USE).get("model_name"))

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
