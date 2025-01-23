""" 
__author__: Lee Jian Hui


USAGE: 
python -m main --simulation # simulating a chat loop againt a list of pre-defined questions
python -m main --benchmark  # benchmark against a list of pre-defined questions
python -m main # normal run as though talking to the chatbot until you type "exit"

FUTURE IMPROVEMENTS:
TODO: add arguments support so the script behavior can be determined by user 
TODO: gracefully handle failures of SQL queries so the model can have graceful fallback

TODO: simulate a system given a client id to:
1. get his transactions within a certain time range
2. classify transactions 

TODO: given a user name tied to client id (assume another new table storing this relationship):
1. get the user's transactions within a certain time range
2. track user's spending habit across all categories and merchants
3. categorize merchants under entertainment, food, etc.


TODO: think about other actions besides SQL querying, can we update the database maybe?
TODO: is it possible for a dynamic max token for the model?
TODO: is it possible for the model to transition into a more QnA style using a RAG pipeline chain
TODO: the benchmark function into an e2e test to cover random user inputs as well (that is logical)
TODO: streaming tokens as an option
TODO: hook up an open source GUI that allows upload of csv files and auto conversion into databases ready to be processed and talked to by the LLM (or other relevant data connectors)
TODO: create a function that downloads and/or post-download-verification across a list of provided model names from hugging face or from locally avaialble .gguf models
TODO: ability to customise further on LLM parameters loaded by llama-cpp
"""


import logging
import os
import time
from typing import Any, List, Optional, Tuple, Union
from llama_cpp import Llama

from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain.schema.cache import BaseCache
from langchain.callbacks.base import Callbacks
from langchain.sql_database import SQLDatabase
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate

from classes import GracefulSQLDatabaseChain
from myprompts import SQLITE_PROMPT
from utils import setup_logger, truncate_conversation_history
from myprompts import prompt_template_generator, _sqlite_prompt1, _sqlite_prompt2, _sqlite_prompt3
import myprompts

MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DATABASE_URL = "sqlite:///data.db" 
MAX_TOKENS = 200
TEMPERATURE = 0.4
CONTEXT_WINDOW_SIZE=8000 # TODO: let this be specified in argvs 
# CONTEXT_WINDOW_SIZE=32768
# CONTEXT_WINDOW_SIZE=4096


logger = setup_logger(__name__, "main.log", level=logging.INFO)



""" 
# TODO: enable to customise all parameters for:

llama_new_context_with_model: n_seq_max     = 1
llama_new_context_with_model: n_ctx         = 32768
llama_new_context_with_model: n_ctx_per_seq = 32768
llama_new_context_with_model: n_batch       = 512
llama_new_context_with_model: n_ubatch      = 512
llama_new_context_with_model: flash_attn    = 0
llama_new_context_with_model: freq_base     = 10000.0
llama_new_context_with_model: freq_scale    = 1
"""
# REF: https://python.langchain.com/docs/how_to/custom_llm/
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


# TODO: RnD on this
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
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        return None


def load_database_connection(database_url: str = DATABASE_URL) -> SQLDatabase: 
    """
    Load the database connection using LangChain's SQLDatabase module.

    Returns:
        SQLDatabase: A connection to the SQL database.
    """
    try:
        db = SQLDatabase.from_uri(database_url)
        logger.info("Database connected successfully.")
        return db
    except Exception as e:
        logger.info(f"Error connecting to database: {e}")
        return None


def create_sql_assistant(
    database: SQLDatabase, 
    llm: CustomLlamaLLM, 
    database_chain_cls: Union[SQLDatabaseChain | SQLDatabaseSequentialChain]=GracefulSQLDatabaseChain 
) -> Union[SQLDatabaseChain | SQLDatabaseSequentialChain]:
    try:
        db_chain = database_chain_cls.from_llm(llm=llm, db=database, verbose=True)

        # 1. To get the raw template string (if applicable)
        # try:
        #     db_chain.llm_chain.prompt = prompt_template if prompt_template is not None else db_chain.llm_chain
        #     logger.info(f"Raw template string: {db_chain.llm_chain}")
        # except AttributeError:
        #     logger.info("This prompt does not have a `template` attribute.")

        logger.info("Banking assistant created successfully.")
        return db_chain
    except Exception as e:
        logger.info(f"Error creating banking assistant: {e}")
        return None

def chat_loop(
    banking_assistant: SQLDatabaseChain, 
    prompt_template: PromptTemplate, 
    max_context_window: int, 
    max_tokens_for_template: int, 
    max_tokens_for_history: int,
    simulated: bool = False, 
    questions: List[str] = None
):
    """
    Chat loop for interacting with the banking assistant, ensuring the prompt template is preserved.

    Args:
        banking_assistant (SQLDatabaseChain): The banking assistant instance.
        prompt_template (PromptTemplate): The fixed prompt template.
        max_context_window (int): The maximum context window size.
        max_tokens_for_template (int): The token count for the fixed prompt template.
        max_tokens_for_history (int): The maximum token allowance for conversation history.
        simulated (bool): Whether to use a simulated chat with predefined questions.
        questions (List[str]): List of questions for simulated mode.
    """
    print("\nWelcome to the Banking Assistant!")
    print("Type your natural language request below, or type 'exit' to quit.")

    # Initialize conversation history
    conversation_history = ""

    if simulated and questions:
        for question in questions:
            print(f"\nYour Query: {question}")
            try:
                # Update conversation history with the new question
                conversation_history += f"User: {question}\n"

                # Truncate conversation history if it exceeds the token limit
                conversation_history = truncate_conversation_history(
                    conversation_history, 
                    max_tokens_for_history
                )

                # Combine the prompt template and truncated conversation history
                full_prompt = prompt_template.template + "\n\n" + conversation_history
                response = banking_assistant.run(full_prompt)

                # Update the conversation history with the assistant's response
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
                # Update conversation history with the new user input
                conversation_history += f"User: {user_input}\n"

                # Truncate conversation history if it exceeds the token limit
                conversation_history = truncate_conversation_history(
                    conversation_history, 
                    max_tokens_for_history
                )

                # Combine the prompt template and truncated conversation history
                full_prompt = prompt_template.template + "\n\n" + conversation_history
                response = banking_assistant.run(full_prompt)

                # Update the conversation history with the assistant's response
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
        logger.info(f"Raw SQL LLM chain template:\n {prompt.template}")

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



# TODO: fix this to use chat loop
def benchmark_models_with_contexts(
    model_paths: List[str],
    context_sizes: List[int],
    prompt_templates: List[PromptTemplate],
    question_set: List[str],
    output_dir: str = "./benchmark_results",
):
    """
    Benchmark different models with varying context window sizes and prompt templates.

    Args:
        model_paths (List[str]): List of model file paths.
        context_sizes (List[int]): List of context window sizes to test.
        prompt_templates (List[PromptTemplate]): List of prompt templates to test.
        question_set (List[str]): Set of questions for benchmarking.
        output_dir (str): Directory to store benchmark reports.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_path in model_paths:
        for context_size in context_sizes:
            model = Llama(model_path=model_path, n_ctx=context_size)
            llm = CustomLlamaLLM(model)

            for prompt_template in prompt_templates:
                report_lines = []
                start_time = time.time()

                # Create SQLDatabaseChain
                database = load_database_connection()
                sql_llm_chain = create_sql_assistant(database, llm)
                sql_llm_chain.llm_chain.prompt = prompt_template

                # Benchmark questions
                report_lines.append("Benchmark Report")
                report_lines.append("=================")
                report_lines.append(f"Context Window Size: {context_size}")
                report_lines.append(f"Prompt Template: {prompt_template.template}")
                report_lines.append("")

                conversation_history = ""
                for idx, question in enumerate(question_set, 1):
                    report_lines.append(f"Question {idx}: {question}")
                    try:
                        conversation_history += f"User: {question}\n"
                        answer = sql_llm_chain.run(conversation_history)
                        conversation_history += f"Assistant: {answer}\n"
                        report_lines.append(f"Answer {idx}: {answer}")
                    except Exception as e:
                        report_lines.append(f"Error processing question {idx}: {e}")
                        continue

                end_time = time.time()
                total_time = end_time - start_time
                report_lines.append("")
                report_lines.append(f"Time Taken: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

                # Save report to a file
                report_filename = (
                    f"benchmark_model_{os.path.basename(model_path).split('.')[0]}_"
                    f"context_{context_size}_prompt_{prompt_templates.index(prompt_template) + 1}.txt"
                )
                report_filepath = os.path.join(output_dir, report_filename)

                with open(report_filepath, "w") as report_file:
                    report_file.write("\n".join(report_lines))

                logger.info(f"Benchmark saved to {report_filepath}")


def build() -> Union[SQLDatabaseChain | SQLDatabaseSequentialChain]:
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
    sql_llm_chain = create_sql_assistant(database, llama_llm)
    if not sql_llm_chain:
        logger.error("Failed to create SQL LLM chain.")
        return
    # overwrite prompt template
    sql_llm_chain.llm_chain.prompt = SQLITE_PROMPT

    
def benchmark_run():
    # Define models, context sizes, and prompts for benchmarking
    model_paths = ["./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"]
    context_sizes = [4096, 8000, 16384]
    prompt_templates = [
        prompt_template_generator(_sqlite_prompt1),
        prompt_template_generator(_sqlite_prompt2),
        prompt_template_generator(_sqlite_prompt3),
    ]
    question_set = [
        "How many rows are in the 'transactions' table?",
        "Can you filter for transactions with merchant '1INFINITE'?",
        # How much did I spend last week?
        # What is the amount I have spent on Uber in the last 5 months?
    ]

    # Run benchmarking
    benchmark_models_with_contexts(model_paths, context_sizes, prompt_templates, question_set)



if __name__ == "__main__":
    # Test the database context
    # test_database_context(sql_llm_chain, database)


    # TODO: add the ability to parse arugment here, e.g. --simulated --benchmark, 
    # both must be mutually exclusive, if no flags passed here this is a non-simulated run


    # benchmark_run()
    sql_llm_chain = build()
    chat_loop(sql_llm_chain, simulated=False)
