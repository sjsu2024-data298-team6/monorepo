from dotenv import load_dotenv

load_dotenv()

from db.queries import queries
import pandas as pd


def display_datasets():
    datasets = queries().get_datasets()
    return pd.DataFrame(datasets)
