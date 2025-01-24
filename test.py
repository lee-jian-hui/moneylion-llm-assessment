from langchain.llms import OpenAI
from langchain_community.tools.sql_database import SQLDatabase
from langchain.chains.sql_database import SQLDatabaseSequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.prompt import DECIDER_PROMPT, PROMPT

# Step 1: Set up the database connection (SQLite example)
db = SQLDatabase.from_uri("sqlite:///data.db")

# Step 2: Define your LLM
llm = OpenAI(temperature=0.4)

# Step 3: Create the SQLDatabaseSequentialChain
sequential_chain = SQLDatabaseSequentialChain.from_llm(
    llm,
    db,
    query_prompt=PROMPT,  # For querying
    decider_prompt=DECIDER_PROMPT,  # For table decision making
)

# Step 4: Use the chain to process a query
query = "How much did Julia Johnson spend last week?"
result = sequential_chain({"query": query})

# Step 5: Print the result
print(result["result"])
