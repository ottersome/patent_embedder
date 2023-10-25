import logging
import random
from pathlib import Path, PosixPath
from typing import Dict, List
from urllib.parse import quote_plus

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .utils.general import setup_logger

"""
Every Instance will contain:
    1. A source document
    2. One of its (possibly many) 1st-degree references
    3. One of its (possibly many) 2nd-degree references
    4. One of its (possibly many) 3rd-degree references
    5. One of its (possibly many) non-related documents
Each of them will receive an embedding and the external loss will be calculated
so as to create an appropriate margin between the source document and its references
"""


class DocumentDataset(Dataset):
    def __init__(
        self,
        samples: List[List[Tensor]],
    ):
        """
        Args:
            data_path: Path to the dataset
            psql_args: Dict[str, str] = See `DatabaseToDataset`
            encoder_max_length: The maximum length of the encoder we will use for training/production
        """
        self.logger = setup_logger("DocumentDataset", level=logging.DEBUG)
        self.dataset = samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return sel.dataset[idx] where dataset is dataframe
        return self.dataset[idx]


"""
This class will be used to pull data from the database and construct the dataset
args:
---
- psql_args: Dict[str, str] = {
        "host": Host where psql dataset is hosted,
        "database": Database name,
        "user": User name,
        "password": Password,
    }
- encoder_max_length: The maximum length of the encoder we will use for training/production
"""


class DatabaseFactory:
    # Check if cache exists
    def __init__(
        self,
        dir_to_cache: PosixPath,
        psql_args,
        tokenizer: PreTrainedTokenizer,
        encoder_max_length: int = 512,
    ):
        # Setup logger
        self.logger = setup_logger(__name__, level=logging.DEBUG)
        self.psql_args = psql_args
        self.encoder_max_length = encoder_max_length
        self.tokenizer = tokenizer
        # Get absolute path without name of file
        self.path_to_cache = Path(dir_to_cache).parent
        self.train_path = dir_to_cache / "dataset_train.parquet"
        self.val_path = dir_to_cache / "dataset_val.parquet"

    def createDataset(self) -> List[DocumentDataset]:
        if self.train_path.exists() and self.val_path.exists():
            # Load cache
            self.logger.info("Found cache file. Will construct from them")
            self.train, self.val = self._construct_from_cache()
        else:
            self.logger.info("Unable to find cache file. Will construct from scratch")
            self.train, self.val = self._construct_from_strach(self.psql_args)
        return [DocumentDataset(self.train), DocumentDataset(self.val)]

    def _construct_from_cache(self):
        # Read arrow files:
        train = pd.read_parquet(self.train_path).values.tolist()
        val = pd.read_parquet(self.val_path).values.tolist()
        return train, val

    def _construct_from_strach(self, psql_args: Dict[str, str]):
        # Create dbToDT
        dbToDt = DatabaseToDataset(psql_args)

        # We might want to cache this bad boi
        samples = dbToDt.pull_data(
            relationships_tablename=psql_args["relationships_table"],
            samples_text_tablename=psql_args["samples_text"],
        )
        # samples_df.to_csv("samples.csv", sep="|") # 🐛For debugging
        self.logger.info(
            f"Obtained the {len(samples)} text samples. Will no tokenized them"
        )

        tokenized_samples = self._tokenize_samples(samples)
        self.logger.info(
            "Difference of ds after tokenization : %s",
            len(samples) - len(tokenized_samples),
        )

        # Do random split
        self.logger.info("Creating train and validation split")
        random.shuffle(tokenized_samples)
        train_size = int(len(tokenized_samples) * 0.8)
        train_ds = tokenized_samples[:train_size]
        val_ds = tokenized_samples[train_size:]
        # Save to parquet
        self.logger.info(
            "Saving to cache fiels {} and {}".format(self.train_path, self.val_path)
        )
        # TODO: maybe be more flexible with how we write the columns rather than hardcoding them
        train_df = pd.DataFrame(
            train_ds, columns=["og_pubnum", "ref0", "ref1", "ref2", "nonref"]
        )
        val_df = pd.DataFrame(
            val_ds, columns=["og_pubnum", "ref0", "ref1", "ref2", "nonref"]
        )
        train_df.to_parquet(self.train_path)
        val_df.to_parquet(self.val_path)

        return train_ds, val_ds

    def _tokenize_samples(self, samples: List[List[str]]) -> List[List[Tensor]]:
        # Before we save to parquet we want to tokenize the samples
        tokenized_samples = []
        for sample in tqdm(samples, desc="Tokenizing samples"):
            # TODO: verify that this indeed truncates to max size of encoder
            encoded_sample = [
                self.tokenizer.encode(
                    samp,
                    max_length=self.encoder_max_length,
                    truncation=True,
                    padding="max_length",
                )
                for samp in sample
            ]
            tokenized_samples.append(encoded_sample)

        self.logger.info("Tokenized the samples. Will now save them to parquet")

        # Save samples: [[]*n] into a parquet file
        return tokenized_samples


class DatabaseToDataset:
    def __init__(self, psql_args: Dict):
        # Establish psql connection to database
        self.logger = setup_logger(__name__, level=logging.DEBUG)
        self.psql_args = psql_args
        try:
            self.engine = create_engine(
                f"postgresql://{psql_args['user']}:{quote_plus(psql_args['password'])}@{psql_args['host']}:{psql_args['port']}/{psql_args['database']}"
            )
            self.conn = self.engine.connect()
        # Catch OperationalError:
        except psycopg2.OperationalError as e:
            self.logger.error("❌ Unable to connect to database:" + str(e))
            exit(-1)

    def _obtain_txt_data_for_row(self, row, table_name):
        # Query the text table for the text of the sample_id
        to_return = []
        for sample_id in row:
            query = f"SELECT title,abstract FROM {table_name} WHERE publication_number = '{sample_id}'"
            res = pd.read_sql(query, self.conn)
            # res.dropna(axis=0, how="any", inplace=True)
            title = res.iloc[0]["title"]
            abstract = res.iloc[0]["abstract"]
            if res.size == 0:
                # We have no text for this sample_id
                self.logger.error(
                    "No matching row in text data for publication number %s",
                    sample_id,
                )
                return None
            if (
                title == None
                or title == ""
                or abstract == None
                or abstract == ""
                or title == "NaN"
                or abstract == "NaN"
            ):
                return None
            # Ensure title + abstract is less than encoder_max_length
            to_return += ["<title>: " + title + "[SEP]<abstract>: " + abstract]
        return to_return

    def pull_data(self, relationships_tablename, samples_text_tablename: str):
        query = f"""SELECT * FROM {relationships_tablename}
            WHERE array_length(ref0,1) < 12
            AND array_length(ref1,1) < 400
            AND array_length(ref2,1) < 1000
            ORDER BY og_pubnum ASC;
        """  # TODO: not a big fan of this way of filtering, well have to fix it later
        try:
            relationships: pd.DataFrame = pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.logger.error("❌ Unable to query database:" + str(e))
            exit(-1)
        # Remove all None values from the dataframe
        relationships.dropna(axis=0, how="any", inplace=True)
        # relationships.to_csv("relationships.csv", sep="|") # 🪲
        samples = []

        # From there we will select just a few to build the dataset (atleast for now)
        dropping_count = 0
        for _, row_series in tqdm(relationships.iterrows(), desc="Going through rows"):
            # Create a list of col_num elements
            i = 0
            stop = False
            while (
                i < 10 and stop == False
            ):  # Have at maximum three samples for each row
                try:
                    # Check that all columns (which are list) are not empty
                    bool_list = [len(column) != 0 for column in row_series.iloc[1:]]
                    if not all(bool_list):
                        stop = True
                        continue
                    # Create sample ids which will we loop through
                    samples_ids = [row_series.iloc[0]] + [
                        column.pop() for column in row_series.iloc[1:]
                    ]
                    # Pop will give us IndexError when it can pop no more
                    # If we managed to construct the Ids (without exception) it means we can get text data, unless...
                    sample = self._obtain_txt_data_for_row(
                        samples_ids, samples_text_tablename
                    )
                    # ... unless we receive a `None` meaning that the text data is missing for one or more of the samples
                    if sample == None:
                        continue
                    # If we reach here, we have a sample, we add to the list and increase i
                    samples.append(sample)
                    i += 1

                # Catch Index Error:
                except IndexError:
                    dropping_count += 1
                    self.logger.error("IndexError: %s", dropping_count)
                    exit(-1)
        # At this point we have all the samples indices we need
        self.logger.info(
            f"Formed the index dataset with total size {len(samples)}. Dropped {dropping_count} samples"
        )
        return samples
