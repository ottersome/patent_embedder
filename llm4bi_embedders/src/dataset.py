import logging
import random
from pathlib import Path, PosixPath
from typing import Dict, Generator, List, Tuple
from urllib.parse import quote_plus

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
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


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        psql_args,
        data_path,
        tokenizer,
        encoder_max_length,
        device,
    ):
        super().__init__()
        self.logger = setup_logger(__name__, level=logging.INFO)
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.device = device
        self.psql_args = psql_args
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length

    def prepare_data(self):
        self.logger.info("📂Creating Dataset")
        datasetFactory = DatabaseFactory(
            dir_to_cache=PosixPath(self.data_path),
            psql_args=self.psql_args,
            tokenizer=self.tokenizer,
            encoder_max_length=self.encoder_max_length,
        )
        self.train_dataset, self.val_dataset = datasetFactory.createDataset(self.device)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # type: ignore
            num_workers=12,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,  # type: ignore
            num_workers=12,
        )


class DocumentDataset(Dataset):
    def __init__(
        self,
        samples: List[List[Tensor]],
    ):
        """
        Args:
            data_path: Path to the dataset
            samples: List of rows. Each rows contains n lists. Each list being the tokens corresponding to title and abstract of each patent.
            device: device to put the tensors on
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
        self.logger = setup_logger(__name__, level=logging.INFO)
        self.psql_args = psql_args
        self.encoder_max_length = encoder_max_length
        self.tokenizer = tokenizer
        # Get absolute path without name of file
        self.path_to_cache = Path(dir_to_cache).parent
        self.train_path = dir_to_cache / "dataset_train.parquet"
        self.val_path = dir_to_cache / "dataset_val.parquet"

    def createDataset(
        self, device: torch.device
    ) -> Tuple[DocumentDataset, DocumentDataset]:
        if self.train_path.exists() and self.val_path.exists():
            # Load cache
            # TODO: MAybe make this less clunky. Feels like im doing redundant calls
            self.logger.info("Found cache file. Will construct from them")
            self.train, self.val = self._construct_from_cache()
            # Take List[List[array-like]] and transform it into List[List[Tensor]]
            self.train = [
                [torch.Tensor(np.array(doc)).to(torch.long) for doc in sample]
                for sample in self.train
            ]
            self.val = [
                [torch.Tensor(np.array(doc)).to(torch.long) for doc in sample]
                for sample in self.val
            ]

        else:
            # TODO: check if I have to pass return here to Tensors just like I do when I
            # construct from cache
            self.logger.info("Unable to find cache file. Will construct from scratch")
            self.train, self.val = self._construct_from_strach(self.psql_args)

        return (DocumentDataset(self.train), DocumentDataset(self.val))

    def _construct_from_cache(self) -> Tuple[List, List]:
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
        columns = (
            ["og_pubnum"]
            + [f"ref{i}" for i in range(len(train_ds[0]) - 2)]
            + ["nonref"]
        )
        train_df = pd.DataFrame(train_ds, columns=columns)
        val_df = pd.DataFrame(val_ds, columns=columns)
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
        from sqlalchemy import create_engine

        self.logger = setup_logger(
            __name__, "DatabaseToDataset.log", level=logging.INFO
        )
        self.psql_args = psql_args

        self.engine = create_engine(
            f"postgresql://{psql_args['user']}:{quote_plus(psql_args['password'])}@{psql_args['host']}:{psql_args['port']}/{psql_args['database']}"
        )
        self.conn = self.engine.connect()
        # Catch OperationalError:

        self.avg_samples_extracted_from_row = 0  # 🪲
        self.tot_samples_added = 0  # 🪲

    def _obtain_txt_data_for_row(self, row, table_name):
        # Query the text table for the text of the sample_id
        to_return = []
        to_remove = []
        for sample_id in row:
            query = f"SELECT title,abstract FROM {table_name} WHERE publication_number = '{sample_id}'"
            res = pd.read_sql(query, self.conn)
            # res.dropna(axis=0, how="any", inplace=True)
            # Check
            if res.empty:
                # We have no text for this sample_id
                self.logger.debug(
                    "No matching row in text data for publication number %s",
                    sample_id,
                )
                to_remove.append(sample_id)
                continue
            title = res.iloc[0]["title"]
            abstract = res.iloc[0]["abstract"]
            self.logger.debug(f"Title {title} and abstract {abstract}")
            if (
                title == None
                or title == ""
                or abstract == None
                or abstract == ""
                or title == "NaN"
                or abstract == "NaN"
            ):
                to_remove.append(sample_id)
                continue
            # Ensure title + abstract is less than encoder_max_length
            to_return += ["<title>: " + title + "[SEP]<abstract>: " + abstract]
        return to_return, to_remove

    def _obtain_txt_data_for_element(self, id, table_name):
        query = (
            f"SELECT title,abstract FROM {table_name} WHERE publication_number = '{id}'"
        )
        res = pd.read_sql(query, self.conn)
        if res.empty:
            return None
        title = res.iloc[0]["title"]
        abstract = res.iloc[0]["abstract"]
        if title == None or title == "" or abstract == None or abstract == "":
            return None
        return "<title>: " + title + "[SEP]<abstract>:" + abstract

    def _pull_one_sample_from_row(
        self, row: pd.Series, samples_text_tablename: str
    ) -> Generator[List[str], None, int]:
        """
        Will take a pandas series as row and get as many samples as possible from it
        """
        # Get column 0 before the others
        col_0_text = self._obtain_txt_data_for_element(
            row["og_pubnum"], samples_text_tablename
        )
        if col_0_text == None:
            self.logger.warning("An og_pubnum was found without text")
            return 0

        ids_in_row_without_text = 0
        pubs_added = 0

        while True:
            text_entry: List[str] = [col_0_text]
            # HACK: remember that for now you are doing simple ref,pos,neg. If you want to dela with chain you have to change this code
            # for col in row[1:]: # For all columns
            for col in row.iloc[[1, -1]]:  # Only includes pos and neg references
                obtained_text = False
                while obtained_text == False:
                    col_list: List[str] = col  # type: ignore
                    # Check Stopping Criterion
                    if len(col_list) == 0:
                        # report statistics
                        column_reports = str([len(c) for c in row[1:]])
                        # HACK: remove once you dont need use for debugging
                        self.logger.debug(
                            f"Statistics Report for og_pubnum {row['og_pubnum']}:\n"
                            f"\tElements taken on each column {pubs_added}\n"
                            f"\tStill remaining {column_reports}.\n"
                            f"\tPubs without text {ids_in_row_without_text}.\n"
                        )
                        return pubs_added
                    id = col_list.pop()

                    text_res = self._obtain_txt_data_for_element(
                        id, samples_text_tablename
                    )
                    if text_res != None:
                        text_entry.append(text_res)
                        obtained_text = True
                    else:
                        ids_in_row_without_text += 1  # 🪲

            pubs_added += 1  # 🪲
            yield text_entry

    def pull_data(self, relationships_tablename, samples_text_tablename: str):
        """
        Main method that will pull data from the database and construct the dataset
        Arguments:
            relationships_tablename: The name of the table in the database that contains the network structure
            samples_text_tablename: The name of the table in the database that contains the text data indexed by publication number
        """
        query = f"""SELECT * FROM {relationships_tablename}
            WHERE array_length(ref0,1) < 12
            AND array_length(ref1,1) < 400
            AND array_length(ref2,1) < 1000
            AND array_length(neg_exs,1) > 0
            ORDER BY og_pubnum ASC;
        """  # TODO: not a big fan of this way of filtering, well have to fix it later
        try:
            relationships: pd.DataFrame = pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.logger.error("❌ Unable to query database:" + str(e))
            exit(-1)

        # Remove all None values from the dataframe
        relationships.dropna(axis=0, how="any", inplace=True)
        # relationships.to_csv("relationships.csv", sep="|")  # 🪲
        samples = []
        # Get columns

        # From there we will select just a few to build the dataset (atleast for now)
        row_i = 0
        tqdm_obj = tqdm(total=len(relationships), desc="Iterating through rows.")
        ####################
        # Get Text
        ####################
        for _, row_series in relationships.iterrows():
            try:
                tqdm_obj.update(1)
                for value in self._pull_one_sample_from_row(
                    row_series, samples_text_tablename
                ):  # When _pull_one_sample_from_row returns none it will stop
                    samples.append(value)
            except StopIteration as e:
                row_i += 1
                result = e.value
                self.avg_samples_extracted_from_row = (
                    (row_i - 1) / row_i
                ) * self.avg_samples_extracted_from_row + (1 / row_i) * result
                self.logger.debug(
                    "Average samples extracted from row: %s",
                    self.avg_samples_extracted_from_row,
                )

        # 🪲 Save this for debugging
        values_df = pd.DataFrame(samples)
        values_df.to_csv("values.csv", sep="|")
        # At this point we have all the samples indices we need
        self.logger.info(f"Formed the index dataset with total size {len(samples)}.")
        return samples
