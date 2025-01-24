""" 
__author__: Lee Jian Hui


USAGE: 
python -m main --simulation # simulating a chat loop againt a list of pre-defined questions
python -m main --benchmark  # benchmark against a list of pre-defined questions
python -m main # normal run as though talking to the chatbot until you type "exit"

FUTURE IMPROVEMENTS:
TODO: implement a mechanism to ask user to wait while tokens are getting processed in terms of a non-streaming implementation
TODO: add arguments support so the script behavior can be determined by user 

TODO: simulate a system given a client id to:
1. get his transactions within a certain time range
2. classify transactions 

TODO: given a user name tied to client id (assume another new table storing this relationship):
1. get the user's transactions within a certain time range
2. track user's spending habit across all categories and merchants
3. categorize merchants under entertainment, food, etc.


TODO: make gracefully failure SQLLLM CHAIN more robust with more variance and configs to the retry mechanism
TODO: think about other actions besides SQL querying, can we update the database maybe?
TODO: is it possible for a dynamic max token for the model?
TODO: is it possible for the model to transition into a more QnA style using a RAG pipeline chain
TODO: the benchmark function into an e2e test to cover random user inputs as well (that is logical)
TODO: streaming tokens as an option
TODO: hook up an open source GUI that allows upload of csv files and auto conversion into databases ready to be processed and talked to by the LLM (or other relevant data connectors)
TODO: create a function that downloads and/or post-download-verification across a list of provided model names from hugging face or from locally avaialble .gguf models

TODO: ability to customise further on LLM parameters loaded by llama-cpp:
llama_new_context_with_model: n_seq_max     = 1
llama_new_context_with_model: n_ctx         = 32768
llama_new_context_with_model: n_ctx_per_seq = 32768
llama_new_context_with_model: n_batch       = 512
llama_new_context_with_model: n_ubatch      = 512
llama_new_context_with_model: flash_attn    = 0
llama_new_context_with_model: freq_base     = 10000.0
llama_new_context_with_model: freq_scale    = 1

TODO: find out how to use ollama to try out deepseek's reasoning model
"""


import logging
import os
import time
from typing import Any, List, Optional, Tuple, Type, Union
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
from mydatabase import initialize_database
from myprompts import DEFAULT_SQLITE_PROMPT
from utils import BenchmarkReport, setup_logger, truncate_conversation_history
from myprompts import prompt_template_generator, _sqlite_prompt1, _sqlite_prompt2, _sqlite_prompt3
import myprompts
from configs import DATABASE_URL, DEFAULT_CHAT_OUTPUT_FILEPATH, DEFAULT_MODEL_PATH


DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE=0.4
DEFAULT_CONTEXT_WINDOW_SIZE=4096 # TODO: let this be specified in argvs 
ALLOWED_WINDOW_SIZES=[2000,4000,8000,16000]
logger = setup_logger(__name__, "main.log", level=logging.INFO)




# TODO: RnD on this
# Mock BaseCache implementation
class SimpleCache(BaseCache):
    def lookup(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


# Ensure SQLDatabaseChain is fully defined
SQLDatabaseChain.model_rebuild()


def load_local_model(
    model_path=DEFAULT_MODEL_PATH, 
    context_window_size: int=DEFAULT_CONTEXT_WINDOW_SIZE
) -> Llama:
    """
    Load the Mistral-7B-GGUF model using llama-cpp-python.

    Args:
        model_path (str): Path to the GGUF model file.

    Returns:
        Llama: Loaded model instance.
    """
    try:
        model = Llama(model_path=model_path, n_ctx=context_window_size)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        return None


# REF: https://python.langchain.com/docs/how_to/custom_llm/
# Custom LLM Wrapper for llama_cpp

class CustomLlamaLLM(LLM):
    def __init__(
        self,
        model: Llama, 
        max_tokens: int = DEFAULT_MAX_TOKENS, 
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Args:
            model (Llama): The Llama model instance.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
        """
        super().__init__()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "llama_cpp"

    def _call(self, prompt: str, stop: list = None, **kwargs: Any) -> str:
        """
        Make a call to the Llama model with the given prompt.

        Args:
            prompt (str): The input prompt.
            stop (list): List of stop sequences.
            **kwargs: Additional arguments.

        Returns:
            str: The model's response.
        """
        response = self._model(
            prompt,
            stop=stop,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response["choices"][0]["text"].strip()


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


# NOTE: intentially separate out from creating LlamaLLM process to be able to swap out different prompt templates without rebuilding LLM instance
def create_llm_chain(
    database: SQLDatabase, 
    llm: CustomLlamaLLM, 
    prompt: Optional[BasePromptTemplate] = None,
    database_chain_cls: Type[Union[SQLDatabaseChain, SQLDatabaseSequentialChain]]=GracefulSQLDatabaseChain,
) -> Union[SQLDatabaseChain, SQLDatabaseSequentialChain]:
    try:
        db_chain = database_chain_cls.from_llm(llm=llm, db=database, prompt=prompt, verbose=True)

        logger.info("Banking assistant created successfully.")
        return db_chain
    except Exception as e:
        logger.info(f"Error creating banking assistant: {e}")
        return None



def chat_loop(
    llm_chain: Union[SQLDatabaseChain, SQLDatabaseSequentialChain], 
    prompt_template: PromptTemplate, 
    max_tokens_for_history: int,
    simulated: bool = False, 
    questions: List[str] = None,
    output_file: Optional[str] = None,
    context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE
):
    """
    Chat loop for interacting with the banking assistant or generating a benchmark report.

    Args:
        llm_chain (Union[SQLDatabaseChain, SQLDatabaseSequentialChain]): The LLM chain instance.
        prompt_template (PromptTemplate): The prompt template.
        max_tokens_for_history (int): The maximum token allowance for conversation history.
        simulated (bool): Whether to simulate questions.
        questions (List[str]): List of questions for simulation.
        output_file (Optional[str]): File to save benchmark results.
        context_window_size (int): Context window size for benchmarking.
    """
    print("\nWelcome to the Banking Assistant!")
    print("Type your natural language request below, or type 'exit' to quit.")

    # Initialize conversation history and benchmark report
    conversation_history = ""
    report = None

    if output_file:
        report = BenchmarkReport(context_window_size, prompt_template.template)

    if simulated and questions:
        for question in questions:
            question = question.strip()  # Normalize input
            print(f"\nYour Query: {question}")
            logger.info(f"Simulated Query: {question}")
            try:
                # Update conversation history
                conversation_history += f"User: {question}\n"

                # # Truncate conversation history if needed
                # conversation_history = truncate_conversation_history(
                #     conversation_history, 
                #     max_tokens_for_history
                # )
                # logger.info(f"\n\nconversation_history:\n{conversation_history}\n\n")

                # Prepare full prompt and log it
                full_prompt = prompt_template.template + "\n\n" + conversation_history
                logger.info(f"Full Prompt Passed to LLM:\n{full_prompt}")

                # Run query
                response = llm_chain.run(full_prompt)

                # Log response
                conversation_history += f"Assistant: {response}\n"
                print("\nQuery Result:")
                print(response)

                if report:
                    report.add_question_and_answer(question, response)
            except Exception as e:
                error_message = f"Error processing query: {e}"
                print(error_message)
                logger.error(error_message)
                if report:
                    report.add_error(question, error_message)
    else:
        while True:
            user_input = input("\nYour Query: ").strip()  # Normalize user input
            logger.info(f"User Query Received: {user_input}")

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            try:
                # Update conversation history
                conversation_history += f"User: {user_input}\n"

                # Truncate conversation history if needed
                # conversation_history = truncate_conversation_history(
                #     conversation_history, 
                #     max_tokens_for_history
                # )

                # Prepare full prompt and log it
                full_prompt = prompt_template.template + "\n\n" + conversation_history
                logger.info(f"Full Prompt Passed to LLM:\n{full_prompt}")

                # Run query
                response = llm_chain.run(full_prompt)

                # Log response
                conversation_history += f"Assistant: {response}\n"
                print("\nQuery Result:")
                print(response)
            except Exception as e:
                error_message = f"Error processing query: {e}"
                print(error_message)
                logger.error(error_message)

    # Save benchmark results if applicable
    if report and output_file:
        report.save_to_file(output_file)
        logger.info(f"Benchmark results saved to {output_file}")


def test_database_context(sql_llm_chain: Union[SQLDatabaseChain, SQLDatabaseSequentialChain], database: SQLDatabase):
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
    context_window_sizes: List[int],
    prompt_templates: List[PromptTemplate],
    question_set: List[str],
    output_dir: str = "./benchmark_results",
) -> None:
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
        for context_window_size in context_window_sizes:
            llama_llm = build_llama_llm(context_window_size=context_window_size)

            # model = Llama(model_path=model_path, n_ctx=context_size)
            # llm = CustomLlamaLLM(model)

            for idx, prompt_template in enumerate(prompt_templates, 1):
                # NOTE: rebuild LLM chain by swapping out prompt template witohut rebuilding llama_llm instance
                llm_chain = build_llm_chain(llama_llm=llama_llm, prompt=prompt_template)
                output_file = os.path.join(
                    output_dir, 
                    f"benchmark_model_{os.path.basename(model_path).split('.')[0]}_"
                    f"context_{context_window_size}_prompt_{idx}.txt"
                )
                
                # Run benchmark using chat_loop
                chat_loop(
                    llm_chain=llm_chain,
                    prompt_template=prompt_template,
                    # max_tokens_for_template=len(prompt_template.template.split()),
                    max_tokens_for_history=context_window_size - len(prompt_template.template.split()),
                    simulated=True,
                    questions=question_set,
                    output_file=output_file,
                    context_window_size=context_window_size
                )


def build_llama_llm(
    model_path: str = DEFAULT_MODEL_PATH, 
    context_window_size: int=DEFAULT_CONTEXT_WINDOW_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: float = DEFAULT_MAX_TOKENS,
) -> Optional[CustomLlamaLLM]:
    model = load_local_model(model_path=model_path, context_window_size=context_window_size)
    if not model:
        logger.error("Model failed to load.")
        return
    return CustomLlamaLLM(model, max_tokens=max_tokens, temperature=temperature)



# TODO: add support to load non-local models in the future from hugging face or ollama directly
def build_llm_chain(
        llama_llm: CustomLlamaLLM,
        prompt: Optional[BasePromptTemplate] = None,
        sql_chain_cls: Type[Union[SQLDatabaseChain, SQLDatabaseSequentialChain]] = GracefulSQLDatabaseChain,
    ) -> Union[SQLDatabaseChain, SQLDatabaseSequentialChain]:
    # Load the model

    # Load the database connection
    database = load_database_connection()
    if not database:
        logger.error("Database connection failed.")
        return

    # Create the banking assistant
    sql_llm_chain = create_llm_chain(database=database, llm=llama_llm, prompt=prompt, database_chain_cls=sql_chain_cls)
    if not sql_llm_chain:
        logger.error("Failed to create SQL LLM chain.")
        return
    
    # overwrite prompt template
    sql_llm_chain.llm_chain.prompt = DEFAULT_SQLITE_PROMPT

    return sql_llm_chain

    
def benchmark_run() -> None:
    # Define models, context sizes, and prompts for benchmarking
    model_paths = ["./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"]
    context_sizes = ALLOWED_WINDOW_SIZES
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
    benchmark_models_with_contexts(
        model_paths, 
        context_sizes, 
        prompt_templates, 
        question_set
    )

def main_run_loop():
    # choose sqllite prompt 3 from my prompt
    context_window_size = DEFAULT_CONTEXT_WINDOW_SIZE
    llama_llm = build_llama_llm(context_window_size=context_window_size)
    sql_llm_chain = build_llm_chain(llama_llm, prompt=DEFAULT_SQLITE_PROMPT, sql_chain_cls=SQLDatabaseChain)
    chat_loop(
        llm_chain=sql_llm_chain, 
        prompt_template=DEFAULT_SQLITE_PROMPT,
        max_tokens_for_history=context_window_size - len(DEFAULT_SQLITE_PROMPT.template.split()),
        simulated=False,
        output_file=DEFAULT_CHAT_OUTPUT_FILEPATH
    )


if __name__ == "__main__":
    # Initialize the SQLite database
    initialize_database()
    # Test the database context
    # test_database_context(sql_llm_chain, database)


    # TODO: add the ability to parse arugment here, e.g. --simulated --benchmark, 
    # both must be mutually exclusive, if no flags passed here this is a non-simulated run


    benchmark_run()
    # main_run_loop()