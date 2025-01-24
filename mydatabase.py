import os
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, MetaData, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy Base and Engine
Base = declarative_base()

class Transaction(Base):
    """
    Schema for the transactions table.
    """
    __tablename__ = 'transactions'

    clnt_id = Column(Integer, nullable=False)
    bank_id = Column(Integer, nullable=False)
    acc_id = Column(Integer, nullable=False)
    txn_id = Column(Integer, primary_key=True)
    txn_date = Column(Date, nullable=False)
    desc = Column(String, nullable=True)
    amt = Column(Float, nullable=False)
    cat = Column(String, nullable=True)
    merchant = Column(String, nullable=True)


class ClientInfo(Base):
    """
    Schema for the client_info table.
    """
    __tablename__ = 'client_info'

    clnt_id = Column(Integer, primary_key=True)
    clnt_name = Column(String, nullable=False)


class SQLiteDB:
    """
    SQLite database class for managing transactions and client_info.
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
        Load a CSV file into a specified table in the database.

        Args:
            csv_file (str): Path to the CSV file.
            table_name (str): Table name to insert data into.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Read the CSV file
        df = pd.read_csv(csv_file)

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
