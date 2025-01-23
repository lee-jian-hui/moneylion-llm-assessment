from llm_providers import build_hf_llm
from db_query import build_db_chain, query_database
from csv_query import build_csv_agent, query_csv

def main():
    # 1. Build an LLM
    llm = build_hf_llm(model_name="tiiuae/falcon-7b-instruct")  # Example instruct model
    
    # 2. Option A: Query a SQLite/Postgres DB
    db_uri = "sqlite:///banking_transactions.db"
    db_chain = build_db_chain(llm, db_uri, verbose=True)
    question_db = "What is the total sum of 'amt' in the transactions table for July 2023?"
    db_answer = query_database(db_chain, question_db)
    print("DB Answer:", db_answer)
    
    # 3. Option B: Query a CSV
    csv_path = "transactions.csv"
    csv_agent = build_csv_agent(llm, csv_path, verbose=True)
    question_csv = "How much did I spend on Uber last month?"
    csv_answer = query_csv(csv_agent, question_csv)
    print("CSV Answer:", csv_answer)

if __name__ == "__main__":
    main()
