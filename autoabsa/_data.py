import pandas as pd


class DataHandler:
    def __init__(self, docs) -> None:
        self.docs = docs

    @staticmethod
    def load_from_json(path):
        df = pd.read_json(path)
        return df
