import logging
import os
import pandas as pd
from sqlalchemy import (
    DateTime, create_engine, Column, Integer, String, Float, Date, MetaData, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from configs import DATABASE_PATH, DATABASE_URL, TRANSACTION_CSV, CLIENT_INFO_CSV
from datetime import datetime   

from my_logger import GLOBAL_LOGGER

logger = GLOBAL_LOGGER
# SQLAlchemy Base and Engine
Base = declarative_base()

class Transaction(Base):
    """
    Schema for the transactions table.
    """
    __tablename__ = 'transactions'
    # NOTE: temporary workaround because transaction id is not unique
    id = Column(Integer, primary_key=True, autoincrement=True)  # auto-generated unique id

    clnt_id = Column(Integer, nullable=False)
    bank_id = Column(Integer, nullable=False)
    acc_id = Column(Integer, nullable=False)
    txn_id = Column(Integer, nullable=False)
    txn_date = Column(DateTime, nullable=False)
    desc = Column(String, nullable=True)
    amt = Column(Float, nullable=False)
    cat = Column(String, nullable=True)
    merchant = Column(String, nullable=True)


class ClientInfo(Base):
    """
    Schema for the clients table.
    """
    __tablename__ = 'clients'
    # id = Column(Integer, primary_key=True, autoincrement=True)

    clnt_id = Column(Integer, primary_key=True)
    # clnt_id = Column(Integer, nullable=False)
    clnt_name = Column(String, nullable=False)


# add other connectors in the future or create an abstraction for other connectors using connection URI/URLs

class SQLiteDB:
    """
    SQLite database class for managing transactions and clients.
    """
    def __init__(self, db_path: str = "data.db"):
        """
        Initialize the database connection and create tables if they don't exist.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.Session = sessionmaker(bind=self.engine)

        # Create all tables
        Base.metadata.create_all(self.engine)


    def load_csv_to_table(self, csv_file: str, table_name: str):
        """
        Load a CSV file into a specified table in the database with date conversion.

        Args:
            csv_file (str): Path to the CSV file.
            table_name (str): Table name to insert data into.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Convert txn_date to ISO 8601 format if the table is transactions
        if table_name == "transactions" and "txn_date" in df.columns:
            try:
                # Ensure proper datetime conversion to preserve both date and time
                df["txn_date"] = pd.to_datetime(
                    df["txn_date"], format="%d/%m/%Y %H:%M"
                ).dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to ISO 8601 format with time
            except Exception as e:
                raise ValueError(f"Error converting txn_date: {e}")

        # Validate table name
        if table_name not in Base.metadata.tables:
            raise ValueError(f"Table '{table_name}' does not exist in the database schema.")

        # Load data into the database
        with self.engine.connect() as connection:
            df.to_sql(table_name, con=connection, if_exists="append", index=False)
        print(f"Data from {csv_file} has been successfully loaded into '{table_name}' table.")


        def query_table(self, table_name: str):
            """
            Query all data from a table.

            Args:
                table_name (str): The name of the table to query.

            Returns:
                List of rows from the table.
            """
            if table_name not in Base.metadata.tables:
                raise ValueError(f"Table '{table_name}' does not exist in the database schema.")

            with self.Session() as session:
                table = Base.metadata.tables[table_name]
                result = session.query(table).all()
            return result

        def get_schema(self, table_name: str):
            """
            Retrieve schema information for a given table.

            Args:
                table_name (str): The name of the table.

            Returns:
                List of column names and types.
            """
            if table_name not in Base.metadata.tables:
                raise ValueError(f"Table '{table_name}' does not exist in the database schema.")

            table = Base.metadata.tables[table_name]
            schema = [(col.name, str(col.type)) for col in table.columns]
            return schema


def initialize_database(db_path: str = DATABASE_PATH, transaction_csv: str = TRANSACTION_CSV, client_csv: str = CLIENT_INFO_CSV):
    """
    Initialize the SQLite database with tables and populate data from CSV files.

    Args:
        db_path (str): Path to the SQLite database file.
        transaction_csv (str): Path to the transaction CSV file.
        client_csv (str): Path to the client info CSV file.
    """
    # Initialize the SQLiteDB instance
    sqlite_db = SQLiteDB()

    # Load data into the database
    try:
        sqlite_db.load_csv_to_table(transaction_csv, "transactions")
        logger.info(f"Loaded data from {transaction_csv} into 'transactions' table.")
    except Exception as e:
        logger.error(f"Error loading transactions CSV: {e}")

    try:
        sqlite_db.load_csv_to_table(client_csv, "clients")
        logger.info(f"Loaded data from {client_csv} into 'clients' table.")
    except Exception as e:
        logger.error(f"Error loading client info CSV: {e}")



if __name__ == "__main__":
    sqlite_db = SQLiteDB()
    sqlite_db.load_csv_to_table("test_transactions.csv", "transactions")
