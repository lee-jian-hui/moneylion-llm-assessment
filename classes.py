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

from sqlalchemy.orm import DeclarativeBase
from mydatabase import Base
from my_logger import GLOBAL_LOGGER


INTERMEDIATE_STEPS_KEY = "intermediate_steps"
SQL_QUERY = "SQLQuery:"
SQL_RESULT = "SQLResult:"

logger = GLOBAL_LOGGER


# class CustomSQLDatabaseChain(SQLDatabaseChain):
#     def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
#         # Access the filled prompt
#         filled_prompt = self.llm_chain.prompt.format(**inputs)
#         logger.info(f"Filled Prompt:\n{filled_prompt}")
#         return super()._call(inputs, run_manager)

# NOTE: trying to overwrite the table info passed into the chain, as well as opportunity to log internal workings
class CustomSQLDatabaseChain(SQLDatabaseChain):
    def _call(self, inputs: dict, run_manager=None):
        # Extract the table_info from the database
        table_names_to_use = inputs.get("table_names_to_use")
        original_table_info = self.database.get_table_info(table_names=table_names_to_use)

        # Modify the table_info here (e.g., inject descriptions or custom formatting)
        modified_table_info = original_table_info + self._generate_extra_table_info() + "\n"
        
        # Replace the original table_info with the modified one
        inputs["table_info"] = modified_table_info

        # Log the modified table_info (optional)
        self.verbose and print(f"Modified Table Info:\n{modified_table_info}")

        # Call the original predict method
        return super()._call(inputs, run_manager)


    def _generate_extra_table_info(
        self,
        base: DeclarativeBase = Base,
        include_sample_data: bool = False,
        sample_rows: int = 3
    ) -> str:
        """
        Generate table information dynamically, injecting descriptions and optionally including sample data.

        Args:
            base (DeclarativeBase): SQLAlchemy Base containing table definitions.
            include_sample_data (bool): Whether to include sample data in the table info.
            sample_rows (int): Number of sample rows to include if `include_sample_data` is True.

        Returns:
            str: Formatted table info string with descriptions and sample data.
        """
        return generate_table_info_from_orm_models(
            base=base,
            include_sample_data=include_sample_data,
            sample_rows=sample_rows
        )



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




from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.schema import CreateTable
from sqlalchemy.types import NullType
from typing import List, Optional
from langchain_community.utilities.sql_database import SQLDatabase


class CustomizableSQLDatabase(SQLDatabase):
    def __init__(
        self,
        *args,
        include_samples: bool = False,
        include_metadata: bool = True,
        base: Optional[DeclarativeBase] = None,
        sample_rows: int = 3,
        **kwargs
    ):
        """
        Custom SQLDatabase that allows optional inclusion of samples and metadata.

        Args:
            include_samples (bool): Whether to include sample rows in the table info.
            include_metadata (bool): Whether to include metadata from SQLAlchemy ORM models.
            base (Optional[DeclarativeBase]): SQLAlchemy Base containing ORM models.
            sample_rows (int): Number of sample rows to include.
        """
        super().__init__(*args, **kwargs)
        self.include_samples = include_samples
        self.include_metadata = include_metadata
        self.base = base
        self.sample_rows = sample_rows

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """
        Custom implementation of `get_table_info`, reusing logic from SQLDatabase and adding
        optional inclusion of sample rows and ORM metadata.

        Args:
            table_names (Optional[List[str]]): Specific table names to retrieve info for.

        Returns:
            str: Customized table information.
        """
        # Reuse the parent class logic to handle table names and reflection
        all_table_names = self.get_usable_table_names() if table_names is None else table_names
        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        to_reflect = set(all_table_names) - set(metadata_table_names)
        if to_reflect:
            self._metadata.reflect(
                views=self._view_support,
                bind=self._engine,
                only=list(to_reflect),
                schema=self._schema,
            )

        # Prepare list of tables
        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # Reuse the parent logic for column and index info
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"

            has_extra_info = (
                self._indexes_in_table_info or self.include_samples
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self.include_samples:
                table_info += f"\n{self._get_sample_rows_custom(table)}\n"
            if has_extra_info:
                table_info += "*/"

            tables.append(table_info)

        # Optionally add metadata from ORM models
        # if self.include_metadata and self.base:
        #     orm_table_info = self._get_orm_metadata(all_table_names)
        #     if orm_table_info:
        #         tables.append(orm_table_info)
        tables.append(generate_table_info_from_orm_models(Base, include_sample_data=False))

        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _get_sample_rows_custom(self, table) -> str:
        """
        Custom implementation for fetching sample rows, reusing the parent logic.

        Args:
            table (Table): SQLAlchemy table object.

        Returns:
            str: Sample rows as a formatted string.
        """
        return self._get_sample_rows(table)

    # def _get_orm_metadata(self, table_names: List[str]) -> str:
    #     """
    #     Include ORM metadata if the DeclarativeBase is provided.

    #     Args:
    #         table_names (List[str]): Table names to include in the metadata.

    #     Returns:
    #         str: Formatted metadata information.
    #     """
    #     if not self.base:
    #         return ""

    #     metadata_info = []
    #     for table_name, table in self.base.metadata.tables.items():
    #         if table_name in table_names:
    #             metadata_info.append(f"Table {table_name}:")
    #             for column in table.columns:
    #                 col_info = f"  {column.name} {column.type}"
    #                 if "description" in column.info:
    #                     col_info += f"  # {column.info['description']}"
    #                 metadata_info.append(col_info)
    #             metadata_info.append("")  # Add blank line after each table

    #     return "\n".join(metadata_info).strip()




def generate_table_info_from_orm_models(
    base: DeclarativeBase = Base,
    include_sample_data: bool = False,
    sample_rows: int = 3
) -> str:
    """
    Generate table information dynamically from SQLAlchemy ORM models,
    including column descriptions and optionally sample data.

    Args:
        base (DeclarativeBase): SQLAlchemy Base containing table definitions.
        include_sample_data (bool): Whether to include sample data from tables.
        sample_rows (int): Number of sample rows to include if `include_sample_data` is True.

    Returns:
        str: Formatted table info string.
    """
    table_info = []
    engine = base.metadata.bind  # Get the engine bound to the base

    if not engine:
        raise ValueError("SQLAlchemy Base is not bound to an engine. Please bind it first.")

    for table_name, table in base.metadata.tables.items():
        table_info.append(f"Table {table_name}:")

        # Iterate through columns and include descriptions
        for column in table.columns:
            column_desc = f"  {column.name} {column.type}"
            
            # Include description from the 'info' attribute if present
            if "description" in column.info:
                column_desc += f"  # {column.info['description']}"
            
            table_info.append(column_desc)

        # Optionally include sample data
        if include_sample_data:
            with engine.connect() as connection:
                try:
                    sample_query = f"SELECT * FROM {table_name} LIMIT {sample_rows}"
                    result = connection.execute(sample_query).fetchall()
                    if result:
                        table_info.append(f"\n  Sample data from {table_name}:")
                        for row in result:
                            table_info.append(f"    {dict(row)}")
                except Exception as e:
                    table_info.append(f"  Unable to fetch sample data: {e}")

        table_info.append("")  # Add a blank line after each table

    return "\n".join(table_info).strip()


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