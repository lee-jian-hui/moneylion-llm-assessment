
# PROJECT DETAILS
- this project is built on top of langchain and llama-cpp leveraging mainly mistral's GGUF models and can be extensible to support other models
- the decision to utilize running .GGUF models on top of llama-cpp is purely due to a resource constraint on my host machine and this decision could be factored in similarly if you were to work on resource constraint devices.

# PREREQUISITES

## Data Files:
- data.csv: containing transaction data
- client.csv: trivial table containingclient id and client name columns
- assumes a .gguf model is downloaded, for reference, the model that this repository uses extensively: 
  - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  - https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF
  - specifically: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
  - `curl -L -o models/mistral-7b-instruct-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf` to download using curl command on Linux
## Environment 


- you must have a virtual environment package manager like conda, poetry, etc. installed and setup properly
- [OPTIONAL | RECOMMENDED] development on my end is done on WSL-Linux environment on Ubuntu22.04 distro and is recommended to follow as well.
- assumes a database `data.db` already loaded from a given `data.csv` file
  - there is legacy code that could be referred from `archive/backup.py` to achieve this
- assumes a minimum amount of RAM to run the GGUF model (~around 5 GB!)

- [OPTIONAL] have a .env file in the same directory as`main.py`, example:
```
MODEL_NAME=xxx
HF_AUTH_TOKEN=xxx
OPENAI_TOKEN=xxx
```

# Installation
```
conda activate yourenv # or replace this with other pakcage manager to activate your virtual env
pip install -r requirements.txt
```


# USAGE


# JUSTIFICATIONS / THOUGHT PROCESS  / CONSIDERATIONS
- a simple but effective brute force method to the given problem statement of building an LLM assistant tasked with helping users with enqueries regarding their personal transactions will be to first query all of the relevant users' personal data after performing an SQL query on the database (moneylion's transaction data) and then feed it to the LLM to perform summarisation and analysis for the users' data. However:
  - this violates data sensivity if we were to dump the data in the context window of a non-locally-hosted LLM
  - the amount of data needed to dump into the LLM can get very large easily especially if the data is complex and requires a lot of JOIN operations
  - dumping the data before determining what kind of data is needed by the LLM by forming a query first can be fast, but very expensive in terms of LLM compute tokens
- by using coding extensively in the implementation of the LLM pipeline, I can also define graceful fallback mechanisms and hook onto other potential backend services that might prove to be useful in a microservices architecture pattern as well!
  - using message queues could be great for a very heavy backend load system and this could be paired with the LLM as a future consideration!
  - 




# TODO: write justificatijon about why i tried to inherit the sql database chain from lagnchain for a retry mechanism on sql query failures even though there is a trivial query chcker mechanism in place already

# TODO: one final run throgujupyter notebook# 
# TODO: write about coming from a backend heavy background with interest in ML /AI/LLM , this code is aimed to serve as a strong foundation to fine tune different configurations of the LLM chain that leads to different generation behaviours for the final answer, this is understand what kof parameters work best if for example we are bound by certain parameter constraints , the most straightfoward one being hardware constraints in my case. We can also have other constraints such as not being able to use state of the art closed-sourced models like GPT-4 which is one of the most powerful for NL2SQL generation.

# one would find the approach i explored is more on the coding-side of things rather than heavy prompt engineering. This is because yet again, this is an open ended challenege and the current prompt engineering techniques for NL2SQL generation might work with a snmall number of tables, one would find prompt engineering techniques becomning harder to maintain as databases as systems get larger and more complex.
# an example is that oif there are too many databases and tit is an overall complex system to query from, few shot prompting in NL2SQL might not prove to be worth it, as there could be many different NL2SQL patterns wihich make one argue that it would be far better to focus on the LLM chain pipelnine and the overall hosted model's capability. Prompt engineering can only get you so far in a non-reaosning LLM paradigm for the current trend set by unless we employ open-source reasoning models and that perhaps would be a completely different paradigm shift in terms of the whole `[NL2SQL2NL]` pipel

# assuming that I will be working heavily in / with backend in the future integrating LLM capabiltiies into existing backend systems , this is why my approch is heavily tuned towards using coding / deploying chaining pipelines to solve the problem statement

# the code can be also easily extended to encompass more tools in the LLM chain [NL > SQL > NL] to become  [NL > SQL >  .... > NL]

# from my findings when experimenting with distilled version of mistral-7-b which is not as powerful as the full model, i found it prompt engineering techniques to be unreliable as a less powerful model can reason far worse than a full model. This results in a very drastic answer outcome for every change to the prompt template. In the benchmark reports, one will find that the LLM returns the correct 




# models teste
different versions of mistral-6b v0.1,v0.2,v0.3
codellama-34b-instruct.Q3_K_L.gguf (takes way too long to generate the output benchmarks for so i gaveup ~30 mins for a simple prompt)


# after experimenting with more advanced models of mistral-7b the syntax of SQL code looks fine, howevever, it easily fails at tasks requiring data across multiple tables, this leads me to believe aneven more advanced pipeline is needed ,we could consider looking into SQLDatabaseSequentialChain from langchain 

# estimated time taken: ~ 20 hrs / 3 days