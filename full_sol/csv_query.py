import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms.base import LLM

def build_csv_agent(llm: LLM, csv_path: str, verbose: bool = False):
    """
    Build an agent that can query CSV data using a Pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    agent = create_pandas_dataframe_agent(llm, df, verbose=verbose)
    return agent

def query_csv(agent, question: str) -> str:
    """
    Run a natural language query against CSV data.
    """
    answer = agent.run(question)
    return answer
