import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import streamlit as st
import time
import threading

def set_db_config(dash):
    load_dotenv()

    if dash == "F1":
        DB_CONFIG = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('F1_DATABASE')
        }
    elif dash == "Football":
        DB_CONFIG = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('Football_DATABASE')
        }
    elif dash == "Cricket":
        DB_CONFIG = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('Cricket_DATABASE')
        }

    return DB_CONFIG

def run_sql_query(dash, table, query, params=None, retries=3, delay=5):
    for attempt in range(retries):
        try:
            DB_CONFIG = set_db_config(dash)
            connection_string = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
            engine = create_engine(connection_string)

            # Define column names based on table and query
            if table == 'constructors_championship':
                columns = ['Position', 'Team', 'Points', 'Wins']
            elif table == 'drivers_championship':
                columns = ['Position', 'Driver', 'Team', 'Points', 'Wins']
            elif table == 'calendar':
                columns = ['Round', 'GrandPrix', 'Circuit', 'Date',  'Pole Sitter', 'Race Winner']
            elif 'DISTINCT year' in query.upper():
                with engine.connect() as conn:
                    result = conn.execute(text(f"{query}")).fetchall()
                    return [row[0] for row in result]
                # columns = ['year']
            else:
                columns = None

            # Use pd.read_sql for DataFrame output
            df = pd.read_sql(text(query), engine, params=params or {})
            if df.empty:
                logger.warning(f"No data found for {table}")
                return df

            # Convert Decimal to float
            if 'Points' in df.columns:
                df['Points'] = df['Points'].astype(float)
            return df
        except Exception as e:
            logger.error(f"Error running SQL query for {table}: {e}")
            if "database is locked" in str(e) and attempt < retries - 1:
                logger.warning(f"Database is locked in thread {threading.get_ident()}; retrying ({attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                logger.error(f"Error verifying data for {table}: {e}")
                return pd.DataFrame()

def get_available_years(query):
    DB_CONFIG = set_db_config("F1")
    years = set()
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"{query}")).fetchall()
        years = [row[0] for row in result]
    except Exception as e:
            print(f"Error verifying data: {e}")

    return sorted(list(years), reverse=True)

# New: Import numpy for type checking
import numpy as np
import logging

# New: Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# New: Function to preprocess parameters to handle NumPy types
def preprocess_params(params):
    if params:
        converted_params = {}
        for key, value in params.items():
            if isinstance(value, np.integer):
                converted_params[key] = int(value)
            else:
                converted_params[key] = value
        return converted_params
    return params