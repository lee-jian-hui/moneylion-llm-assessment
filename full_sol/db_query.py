from langchain.chains import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.llms.base import LLM
from typing import Optional

def build_db_chain(llm: LLM, db_uri: str, verbose: bool = False):
    """
    Create a chain that can query a database via natural language.
    """
    db = SQLDatabase.from_uri(db_uri)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=verbose, return_intermediate_steps=False)
    return db_chain

def query_database(db_chain: SQLDatabaseChain, question: str) -> str:
    """
    Run a natural language query against the chain. 
    """
    answer = db_chain.run(question)
    return answer
