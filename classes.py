from typing import Dict, Optional, Any, List


from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain.schema.cache import BaseCache
from langchain.callbacks.base import Callbacks
from langchain.sql_database import SQLDatabase
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import DECIDER_PROMPT, PROMPT, SQL_PROMPTS
from langchain.schema import BasePromptTemplate
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate


from my_logger import GLOBAL_LOGGER

INTERMEDIATE_STEPS_KEY = "intermediate_steps"
SQL_QUERY = "SQLQuery:"
SQL_RESULT = "SQLResult:"


logger = GLOBAL_LOGGER

class CustomSQLDatabaseChain(SQLDatabaseChain):
    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        # Access the filled prompt
        filled_prompt = self.llm_chain.prompt.format(**inputs)
        logger.info(f"Filled Prompt:\n{filled_prompt}")
        return super()._call(inputs, run_manager)


class CustomSQLDatabaseSequentialChain(SQLDatabaseSequentialChain):
    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        # Access the filled prompt
        filled_prompt = self.sql_chain.llm_chain.prompt.format(**inputs)
        logger.info(f"Filled Prompt:\n{filled_prompt}")
        return super()._call(inputs, run_manager)


# class CustomSQLDatabaseSequentialChain(SQLDatabaseSequentialChain):
#     def run(self, *args, **kwargs):
#         # Access the filled prompt
#         filled_prompt = self.llm_chain.prompt.format(**kwargs)
#         logger.info(f"Filled Prompt:\n{filled_prompt}")
#         return super().run(*args, **kwargs)


# NOTE: the behvaiour is not robust yet, so more modifications are necessary
class GracefulSQLDatabaseChain(SQLDatabaseChain):
    """A database chain that gracefully handles invalid SQL queries."""

    max_attempts: int = 3  # Maximum number of attempts to reclarify the query

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        attempt = 0
        intermediate_steps: List = []

        while attempt < self.max_attempts:
            try:
                attempt += 1
                _run_manager.on_text(
                    f"\nAttempt {attempt} of {self.max_attempts}", verbose=self.verbose
                )

                input_text = f"{inputs[self.input_key]}\n{SQL_QUERY}"
                _run_manager.on_text(input_text, verbose=self.verbose)

                # If not present, defaults to None (all tables used).
                table_names_to_use = inputs.get("table_names_to_use")
                table_info = self.database.get_table_info(table_names=table_names_to_use)
                llm_inputs = {
                    "input": input_text,
                    "top_k": str(self.top_k),
                    "dialect": self.database.dialect,
                    "table_info": table_info,
                    "stop": ["\nSQLResult:"],
                }
                if self.memory is not None:
                    for k in self.memory.memory_variables:
                        llm_inputs[k] = inputs[k]

                intermediate_steps.append(llm_inputs.copy())  # Log input to SQL generation
                sql_cmd = self.llm_chain.predict(
                    callbacks=_run_manager.get_child(),
                    **llm_inputs,
                ).strip()

                if self.return_sql:
                    return {self.output_key: sql_cmd}

                if not self.use_query_checker:
                    _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
                    intermediate_steps.append(sql_cmd)  # Log SQL query generation
                    intermediate_steps.append({"sql_cmd": sql_cmd})  # Input to execution
                    if SQL_QUERY in sql_cmd:
                        sql_cmd = sql_cmd.split(SQL_QUERY)[1].strip()
                    if SQL_RESULT in sql_cmd:
                        sql_cmd = sql_cmd.split(SQL_RESULT)[0].strip()
                    result = self.database.run(sql_cmd)
                    
                    intermediate_steps.append(str(result))  # Log SQL execution result
                else:
                    # Use Query Checker to validate SQL
                    query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                        template=QUERY_CHECKER, input_variables=["query", "dialect"]
                    )
                    query_checker_chain = LLMChain(
                        llm=self.llm_chain.llm, prompt=query_checker_prompt
                    )
                    query_checker_inputs = {
                        "query": sql_cmd,
                        "dialect": self.database.dialect,
                    }
                    checked_sql_command = query_checker_chain.predict(
                        callbacks=_run_manager.get_child(), **query_checker_inputs
                    ).strip()
                    intermediate_steps.append(checked_sql_command)  # Log checked SQL
                    _run_manager.on_text(
                        checked_sql_command, color="green", verbose=self.verbose
                    )
                    intermediate_steps.append({"sql_cmd": checked_sql_command})  # Input to execution
                    result = self.database.run(checked_sql_command)
                    intermediate_steps.append(str(result))  # Log SQL execution result
                    sql_cmd = checked_sql_command

                _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
                _run_manager.on_text(str(result), color="yellow", verbose=self.verbose)

                # Generate final output
                if self.return_direct:
                    final_result = result
                else:
                    _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                    input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
                    llm_inputs["input"] = input_text
                    intermediate_steps.append(llm_inputs.copy())  # Log input to final answer generation
                    final_result = self.llm_chain.predict(
                        callbacks=_run_manager.get_child(),
                        **llm_inputs,
                    ).strip()
                    intermediate_steps.append(final_result)  # Log final answer
                    _run_manager.on_text(final_result, color="green", verbose=self.verbose)

                chain_result: Dict[str, Any] = {self.output_key: final_result}
                if self.return_intermediate_steps:
                    chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
                return chain_result

            except Exception as exc:
                # Log the error and prepare for clarification
                _run_manager.on_text(
                    f"\nError encountered: {str(exc)}", color="red", verbose=self.verbose
                )
                # Prompt user for clarification
                reclarify_prompt = PromptTemplate(
                    input_variables=["query", "error"],
                    template="Your query '{query}' could not be executed because of the following error: {error}. Could you please rephrase or clarify your question?",
                )
                reclarification_chain = LLMChain(
                    llm=self.llm_chain.llm, prompt=reclarify_prompt
                )
                clarification = reclarification_chain.predict(
                    query=inputs[self.input_key], error=str(exc)
                )
                _run_manager.on_text(
                    f"\nReclarified query: {clarification}", color="yellow", verbose=self.verbose
                )
                inputs[self.input_key] = clarification  # Update the query for retry

        # If all attempts fail, return a graceful error message
        return {
            self.output_key: "Unable to generate a valid query after multiple attempts. Please try again later or contact support."
        }


if __name__ == "__main__":
    # sample usage
    # Create the database chain with graceful fallback
    # graceful_chain = GracefulSQLDatabaseChain.from_llm(
    #     llm=my_llm,
    #     db=SQLDatabase.from_uri("sqlite:///example.db"),
    #     max_attempts=3,
    # )

    # # Run the chain
    # result = graceful_chain.run({"query": "What are the top products by sales?"})
    # print(result["result"])

    pass