
# PROJECT DETAILS
- this project is built on top of langchain and llama-cpp leveraging mainly mistral's GGUF models and can be extensible to support other models
- the decision to utilize running .GGUF models on top of llama-cpp is purely due to a resource constraint on my host machine and this decision could be factored in similarly if you were to work on resource constraint devices.

# PREREQUISITES
- you must have a virtual environment package manager like conda, poetry, etc. installed and setup properly
- [OPTIONAL | RECOMMENDED] development on my end is done on WSL-Linux environment on Ubuntu22.04 distro and is recommended to follow as well.
- assumes a database `data.db` already loaded from a given `data.csv` file
  - there is legacy code that could be referred from `archive/backup.py` to achieve this
- assumes a minimum amount of RAM to run the GGUF model (~around 5 GB!)
- assumes a .gguf model is downloaded, for reference, the model that this repository uses extensively: 
  - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
  - specifically: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
  - `curl -L -o models/mistral-7b-instruct-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf` to download using curl command on Linux
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


# JUSTIFICATION
- a simple but effective brute force method to the given problem statement of building an LLM assistant tasked with helping users with enqueries regarding their personal transactions will be to first query all of the relevant users' personal data after performing an SQL query on the database (moneylion's transaction data) and then feed it to the LLM to perform summarisation and analysis for the users' data. However:
  - this violates data sensivity if we were to dump the data in the context window of a non-locally-hosted LLM
  - the amount of data needed to dump into the LLM can get very large easily especially if the data is complex and requires a lot of JOIN operations
  - dumping the data before determining what kind of data is needed by the LLM by forming a query first can be fast, but very expensive in terms of LLM compute tokens
- by using coding extensively in the implementation of the LLM pipeline, I can also define graceful fallback mechanisms and hook onto other potential backend services that might prove to be useful in a microservices architecture pattern as well!
  - using message queues could be great for a very heavy backend load system and this could be paired with the LLM as a future consideration!
  - 

