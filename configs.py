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