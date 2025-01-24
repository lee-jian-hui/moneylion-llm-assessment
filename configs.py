""" 
Declare constants and extracted vars from os env here
"""

from dotenv import load_dotenv
import os

load_dotenv(".env")


def getenv_and_echo(key, default_val):
    val = os.getenv(key, default_val)
    print(f"{key}: {val}")
    return val


DEFAULT_MODEL_PATH = getenv_and_echo("DEFAULT_MODEL_PATH", "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
DATABASE_PATH=getenv_and_echo("DATABASE_PATH", "data.db" )
DATABASE_URL=getenv_and_echo("DATABASE_URL", "sqlite:///data.db" )
TRANSACTION_CSV=getenv_and_echo("TRANSACTION_CSV", "data.csv")
CLIENT_INFO_CSV=getenv_and_echo("CLIENT_INFO_CSV", "clients.csv")
DEFAULT_CHAT_OUTPUT_FILEPATH=getenv_and_echo("DEFAULT_CHAT_OUTPUT_FILEPATH", "chat_report.txt")

# constants
BENCHMARK_QUES_SET=[
    "How many rows are in the 'transactions' table?",
    "Can you filter for transactions with merchant '1INFINITE'?",
    "How much did Julia Johnson spend last week?",
    "What is the amount Julia Johnson have spent on Uber in the last 5 months?"
]

# Q3 < Q4 < Q5 in terms of mistral quality , larger model， better output
MODEL_PATHS=[
    # "./models/codellama-34b-instruct.Q3_K_L.gguf",
    # "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "./models/mistral-7b-instruct-v0.2-Q4_K_M.gguf",
    "./models/mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
]