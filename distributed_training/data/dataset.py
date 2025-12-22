# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import asyncio
import io
import os
import random
import typing

import aiohttp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import time
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

# Optional: aiobotocore for async S3/R2 operations
try:
    from aiobotocore.session import get_session as get_aio_session
    HAS_AIOBOTOCORE = True
except ImportError:
    HAS_AIOBOTOCORE = False

# Fallback to boto3 for sync operations
try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class SubsetLoader(IterableDataset):
    """
    Base class for data-specific subset loader classes.

    """

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples
        self.num_rows_per_page = 100

        # Buffers
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []
        self.lock = asyncio.Lock()

    async def fetch_data_for_pages(self, pages):
        """
        Set the pages to be used to fill the buffer. Then fetch the page data
        to the buffer.
        """

        self.pages = pages

        # Empty the buffer if it is not.
        self.buffer = []

        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_data_for_page(page, session) for page in self.pages]
            await asyncio.gather(*tasks)

    async def _fetch_data_for_page(self, page, session):
        retry_limit = 10
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = dict(
                dataset=self.name,
                config=config_name,
                split=split,
                offset=page_number,
                limit=self.num_rows_per_page,
            )

            try:
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Prepare the data to append
                    buffer_to_append = []
                    for row in data["rows"]:
                        content = row["row"]["text"]
                        input_ids = self.tokenizer(content, truncation=True)[
                            "input_ids"
                        ]
                        buffer_to_append.extend(input_ids)
                        buffer_to_append.append(self.tokenizer.eos_token_id)

                    async with self.lock:
                        self.buffer.extend(buffer_to_append)
                        self.pages.append((config_name, page_number, split))
                    break  # Success, exit retry loop

            except aiohttp.ClientResponseError:
                attempt += 1
                if attempt < retry_limit:
                    await asyncio.sleep(5)
                else:
                    raise

    def _get_pad_size(self, input_ids):
        """Calculate padding size for a sequence"""
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        return self.sequence_length - remainder if remainder != 0 else 0

    def _refill_padded_buffer(self):
        """
        This methods pulls one page from `self.buffer`, pads it and pushes
        it to the `self.padded_buffer`.
        """
        print(f"\n--- Starting _refill_padded_buffer ---")
        print(
            f"Initial state: buffer size={len(self.buffer)}, padded_buffer size={len(self.padded_buffer)}"
        )

        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            try:
                # search for EOS token index and cut the buffer at it.
                EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
                print(f"Found EOS token at index {EOS_index}")

                input_ids = self.buffer[: EOS_index + 1]
                self.buffer = self.buffer[EOS_index + 1 :]

                self.used_buffer += input_ids

                # Add to padded buffer without the EOS token.
                self.padded_buffer += input_ids[:-1]

                # Calculate and apply padding
                pad_size = self._get_pad_size(input_ids=input_ids[:-1])
                print(
                    f"Adding sequence of length {len(input_ids[:-1])} with padding {pad_size}"
                )

                self.padded_buffer += [self.tokenizer.eos_token_id] * pad_size

            except ValueError:
                print("No EOS token found in buffer!")
                if len(self.buffer) > 0:
                    print(f"Buffer content preview: {self.buffer[:10]}...")
                break

        print(
            f"Final state: buffer size={len(self.buffer)}, padded_buffer size={len(self.padded_buffer)}"
        )
        print("--- Finished _refill_padded_buffer ---\n")

    def __iter__(self):
        return self

    def __next__(self):
        # Check if we have enough tokens for at least one batch
        required_tokens = self.sequence_length * self.batch_size
        if len(self.buffer) < required_tokens:
            raise StopIteration

        batch = []
        labels = []

        for i in range(self.batch_size):
            # Get input sequence and pad if necessary
            current_seq = self.buffer[: self.sequence_length]
            current_seq_len = len(current_seq)

            if current_seq_len != self.sequence_length:
                input_seq = current_seq + [self.tokenizer.eos_token_id] * (
                    self.sequence_length - current_seq_len
                )
            else:
                input_seq = current_seq

            # Get label sequence (shifted by 1) and pad if necessary
            label_seq_raw = self.buffer[1 : self.sequence_length + 1]
            label_seq_len = len(label_seq_raw)

            if label_seq_len != self.sequence_length:
                label_seq = label_seq_raw + [self.tokenizer.eos_token_id] * (
                    self.sequence_length - label_seq_len
                )
            else:
                label_seq = label_seq_raw

            # Add to batch
            batch.append(torch.tensor(input_seq))
            labels.append(torch.tensor(label_seq))

            # Move buffer forward
            self.buffer = self.buffer[self.sequence_length :]

        stacked_batch = torch.stack(batch)
        stacked_labels = torch.stack(labels)

        return stacked_batch, stacked_labels


class DatasetLoader(SubsetLoader):
    name: str = "HuggingFaceFW/fineweb-edu"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 5  # Number of retries
    retry_delay: int = 60  # Seconds to wait between retries
    num_rows_per_page: int = 100

    logger = bt.logging

    @staticmethod
    async def next_pages(
        offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100
    ):
        configs_data = await DatasetLoader.fetch_dataset_configs()
        keys = sorted(configs_data.keys())
        rng = np.random.default_rng(
            hash(seed) & 0xFFFFFFFF
        )  # Create a generator with a seed
        rng.bit_generator.advance(offset)  # Efficiently skip ahead `n` steps
        result = []
        for _ in range(n_pages):
            idx = rng.integers(0, len(keys))
            cfg = keys[idx]
            config = rng.choice(list(configs_data.keys()))
            choice = rng.integers(
                0, configs_data[cfg]["num_rows"] - 1 - num_rows_per_page
            )
            result.append((cfg, int(choice), configs_data[cfg]["split"]))
        return result

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        pages_info=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples
        )

        # Initialize properties
        self.configs_data = None
        self.pages = []
        self.buffer = []
        self.lock = asyncio.Lock()  # For thread-safe operations

    @classmethod
    async def create(
        cls,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        pages_info=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
    ):
        self = cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
        )

        # Fetch dataset configs asynchronously
        self.configs_data = await cls.fetch_dataset_configs()

        if pages_info is not None:
            await self._fetch(pages_info)
        elif self.num_pages:
            await self._fetch_data_to_buffer(self.num_pages)

        return self

    async def _fetch(self, page_info: typing.Tuple[str, int, str], batch_size: int = 5):
        self.pages = list(page_info)

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(self.pages), batch_size):
                batch = self.pages[i : i + batch_size]
                tasks = [
                    self._fetch_data_for_page((config_name, page, split), session)
                    for (config_name, page, split) in batch
                ]
                await asyncio.gather(*tasks)

    async def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        This method sets the `pages` property.
        """
        self.pages = []
        pages_to_fetch = self.get_random_pages(num_pages)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_data_for_page(page, session) for page in pages_to_fetch
            ]
            await asyncio.gather(*tasks)

    async def fetch_data_to_rows(self, num_pages):
        rows = []
        pages_to_fetch = self.get_random_pages(num_pages)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_rows_for_page(page, session) for page in pages_to_fetch
            ]
            results = await asyncio.gather(*tasks)
            for page_rows in results:
                rows.extend(page_rows)

        return rows

    async def _fetch_data_for_page(self, page, session):
        """
        Fetches data asynchronously for a single page, processes it without blocking the event loop,
        and appends the tokenized data to the buffer.

        Args:
            page: A tuple containing the config name, page number, and split.
            session: The HTTP session used for making requests.

        Raises:
            Exception: If the maximum number of retry attempts is exceeded.
        """
        retry_limit = self.retry_limit
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = {
                "dataset": self.name,
                "config": config_name,
                "split": split,
                "offset": page_number,
                "limit": self.num_rows_per_page,
            }

            try:
                # Make an asynchronous HTTP GET request to fetch the data
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = await response.json()

                    # Prepare the data to append
                    buffer_to_append = []

                    # Asynchronously process each row without blocking the event loop
                    tasks = [
                        self._tokenize_content(row["row"]["text"])
                        for row in data["rows"]
                    ]

                    # Gather the tokenized results concurrently
                    row_input_ids = await asyncio.gather(*tasks)

                    # Flatten the list of input IDs and append them to the buffer
                    for input_ids in row_input_ids:
                        buffer_to_append.extend(input_ids)

                    # Safely append the processed data to the shared buffer
                    async with self.lock:
                        self.buffer.extend(buffer_to_append)
                        self.pages.append((config_name, page_number, split))
                    break  # Success, exit retry loop

            except aiohttp.ClientResponseError as e:
                # Handle HTTP client errors with a retry mechanism
                attempt += 1
                if attempt < retry_limit:
                    self.logger.debug(
                        f"Retrying page {page} due to error: {e}. Attempt {attempt} of {retry_limit}"
                    )
                    self.logger.debug(
                        f"Waiting {self.retry_delay * attempt} seconds before retrying..."
                    )
                    await asyncio.sleep(
                        self.retry_delay * attempt
                    )  # Wait before retrying
                else:
                    raise Exception(
                        f"Maximum retry attempts exceeded for page {page}"
                    ) from e

    async def _tokenize_content(self, content):
        """
        Asynchronously tokenizes a string of content using the tokenizer in a separate thread.

        Args:
            content: The text content to be tokenized.

        Returns:
            The list of token IDs for the content, including the EOS token.
        """
        # Offload the CPU-bound tokenization to a thread executor to prevent blocking the event loop
        input_ids = await asyncio.to_thread(
            self.tokenizer.encode,
            content,
            truncation=True,
            max_length=self.sequence_length,
        )
        input_ids.append(self.tokenizer.eos_token_id)
        return input_ids

    async def _fetch_rows_for_page(self, page, session):
        retry_limit = self.retry_limit
        attempt = 0
        while attempt < retry_limit:
            config_name, page_number, split = page

            # Create the request parameters
            params = dict(
                dataset=self.name,
                config=config_name,
                split=split,
                offset=page_number,
                limit=self.num_rows_per_page,
            )

            try:
                async with session.get(self.rows_base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Collect the rows
                    return [row["row"]["text"] for row in data["rows"]]

            except aiohttp.ClientResponseError:
                attempt += 1
                if attempt < retry_limit:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    def get_random_pages(self, num_pages):
        """
        Randomly sample pages.
        A page is a row number of a given split of a given dataset dump.
        """
        pages = []

        for _ in range(num_pages):
            # Choose a random config
            config_name = random.choice(list(self.configs_data.keys()))

            # Choose a random page (row)
            page = random.randint(
                0,
                self.configs_data[config_name]["num_rows"] - 1 - self.num_rows_per_page,
            )

            split = self.configs_data[config_name]["split"]

            pages.append((config_name, page, split))

        return pages

    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple.
        """
        page_names = []

        if hasattr(self, "pages"):
            page_names = [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]

        return page_names

    @staticmethod
    async def fetch_dataset_configs() -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(dataset=DatasetLoader.name)

        attempt = 0
        while attempt < DatasetLoader.retry_limit:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        DatasetLoader.size_base_url, params=params
                    ) as response:
                        response.raise_for_status()

                        data = await response.json()

                        # Extract the configs dict
                        configs_dict = data["size"]["splits"]

                        # Now create a dict with config names (except 'default') as
                        # keys, and the number of rows as values
                        configs_data = {
                            entry["config"]: {
                                "num_rows": entry["num_rows"],
                                "split": entry["split"],
                            }
                            for entry in configs_dict
                            if entry["config"] != "default"
                        }

                        return configs_data

            except aiohttp.ClientResponseError:
                attempt += 1
                if attempt < DatasetLoader.retry_limit:
                    await asyncio.sleep(DatasetLoader.retry_delay)
                else:
                    raise

    @staticmethod
    async def next_pages_async(
        offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100
    ):
        configs_data = await DatasetLoader.fetch_dataset_configs()
        rng = np.random.default_rng(
            hash(seed) & 0xFFFFFFFF
        )  # Create a generator with a seed
        rng.bit_generator.advance(offset)  # Efficiently skip ahead `n` steps
        result = []
        for _ in range(n_pages):
            config = rng.choice(list(configs_data.keys()))
            choice = rng.integers(
                0, configs_data[config]["num_rows"] - 1 - num_rows_per_page
            )
            result.append((str(config), int(choice), configs_data[config]["split"]))
        return result


class R2DatasetLoader(SubsetLoader):
    """
    DatasetLoader that fetches data from Cloudflare R2 instead of HuggingFace API.

    This loader is designed to work with parquet files stored in R2 that contain
    the same data format as HuggingFace's fineweb-edu dataset. The parquet files
    should have a 'text' column containing the training text.

    R2 Configuration:
        The loader reads R2 configuration from environment variables:
        - R2_DATASET_ACCOUNT_ID: Cloudflare account ID
        - R2_DATASET_BUCKET_NAME: Name of the R2 bucket containing dataset
        - R2_DATASET_ACCESS_KEY_ID: Access key for reading the bucket
        - R2_DATASET_SECRET_ACCESS_KEY: Secret key for reading the bucket

    Bucket Structure:
        The bucket should contain parquet files organized by config/shard:
        - {config_name}/{shard_id}.parquet
        - Each parquet file should have a 'text' column with the content

        A metadata file 'configs.json' at the bucket root should describe available
        configs with their row counts and splits.
    """

    # R2 endpoint URL pattern
    R2_ENDPOINT_PATTERN: str = "https://{account_id}.r2.cloudflarestorage.com"

    retry_limit: int = 5
    retry_delay: int = 60
    num_rows_per_page: int = 100

    logger = bt.logging

    # Cache for configs data
    _configs_cache: typing.Dict[str, typing.Dict] = None
    _configs_cache_time: float = 0
    _configs_cache_ttl: float = 300  # 5 minutes

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        pages_info=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        r2_account_id: str = None,
        r2_bucket_name: str = None,
        r2_access_key_id: str = None,
        r2_secret_access_key: str = None,
    ):
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples
        )

        # R2 configuration from parameters or environment
        self.r2_account_id = r2_account_id or os.getenv("R2_DATASET_ACCOUNT_ID")
        self.r2_bucket_name = r2_bucket_name or os.getenv("R2_DATASET_BUCKET_NAME")
        self.r2_access_key_id = r2_access_key_id or os.getenv("R2_DATASET_ACCESS_KEY_ID")
        self.r2_secret_access_key = r2_secret_access_key or os.getenv("R2_DATASET_SECRET_ACCESS_KEY")

        # Validate configuration
        if not all([self.r2_account_id, self.r2_bucket_name, self.r2_access_key_id, self.r2_secret_access_key]):
            raise ValueError(
                "R2 configuration incomplete. Please set R2_DATASET_ACCOUNT_ID, "
                "R2_DATASET_BUCKET_NAME, R2_DATASET_ACCESS_KEY_ID, and R2_DATASET_SECRET_ACCESS_KEY "
                "environment variables or pass them as parameters."
            )

        self.r2_endpoint = self.R2_ENDPOINT_PATTERN.format(account_id=self.r2_account_id)

        # Initialize properties
        self.configs_data = None
        self.pages = []
        self.buffer = []
        self.lock = asyncio.Lock()

    def _get_s3_client(self):
        """Create a boto3 S3 client configured for R2."""
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for R2DatasetLoader. Please install it with: pip install boto3")

        return boto3.client(
            "s3",
            endpoint_url=self.r2_endpoint,
            aws_access_key_id=self.r2_access_key_id,
            aws_secret_access_key=self.r2_secret_access_key,
            region_name="auto",
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=60,
            ),
        )

    @classmethod
    async def create(
        cls,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        pages_info=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        r2_account_id: str = None,
        r2_bucket_name: str = None,
        r2_access_key_id: str = None,
        r2_secret_access_key: str = None,
    ):
        """Factory method to create R2DatasetLoader asynchronously."""
        self = cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_pages=num_pages,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
            r2_account_id=r2_account_id,
            r2_bucket_name=r2_bucket_name,
            r2_access_key_id=r2_access_key_id,
            r2_secret_access_key=r2_secret_access_key,
        )

        # Fetch dataset configs asynchronously
        self.configs_data = await self.fetch_dataset_configs()

        if pages_info is not None:
            await self._fetch(pages_info)
        elif self.num_pages:
            await self._fetch_data_to_buffer(self.num_pages)

        return self

    async def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch dataset configuration from R2.

        Expects a 'configs.json' file at the bucket root with structure:
        {
            "config_name": {
                "num_rows": 123456,
                "split": "train",
                "num_shards": 10,
                "rows_per_shard": 12345
            },
            ...
        }

        If configs.json doesn't exist, it will try to discover configs
        by listing parquet files in the bucket.
        """
        # Check cache
        current_time = time.time()
        if (
            R2DatasetLoader._configs_cache is not None
            and (current_time - R2DatasetLoader._configs_cache_time) < R2DatasetLoader._configs_cache_ttl
        ):
            return R2DatasetLoader._configs_cache

        attempt = 0
        while attempt < self.retry_limit:
            try:
                configs_data = await asyncio.to_thread(self._fetch_configs_sync)
                R2DatasetLoader._configs_cache = configs_data
                R2DatasetLoader._configs_cache_time = current_time
                return configs_data
            except Exception as e:
                attempt += 1
                if attempt < self.retry_limit:
                    self.logger.debug(
                        f"Retrying configs fetch due to error: {e}. Attempt {attempt}"
                    )
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    raise

    def _fetch_configs_sync(self) -> typing.Dict[str, typing.Dict]:
        """Synchronously fetch configs from R2."""
        import json
        from botocore.exceptions import ClientError

        s3 = self._get_s3_client()

        try:
            # Try to get configs.json first
            response = s3.get_object(Bucket=self.r2_bucket_name, Key="configs.json")
            configs_data = json.loads(response["Body"].read().decode("utf-8"))
            return configs_data
        except ClientError as e:
            # If configs.json doesn't exist, discover configs by listing bucket
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return self._discover_configs_sync(s3)
            raise

    def _discover_configs_sync(self, s3) -> typing.Dict[str, typing.Dict]:
        """
        Discover available configs by listing parquet files in the bucket.
        Assumes structure: {config_name}/{shard_id}.parquet
        """
        configs_data = {}
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.r2_bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    parts = key.rsplit("/", 1)
                    if len(parts) == 2:
                        config_name = parts[0]
                        if config_name not in configs_data:
                            configs_data[config_name] = {
                                "num_rows": 0,
                                "split": "train",
                                "shards": [],
                            }
                        configs_data[config_name]["shards"].append(key)

        # Estimate row counts by reading first shard of each config
        for config_name, config_info in configs_data.items():
            if config_info["shards"]:
                first_shard = config_info["shards"][0]
                try:
                    response = s3.get_object(Bucket=self.r2_bucket_name, Key=first_shard)
                    parquet_data = response["Body"].read()
                    table = pq.read_table(io.BytesIO(parquet_data))
                    rows_per_shard = len(table)
                    config_info["num_rows"] = rows_per_shard * len(config_info["shards"])
                    config_info["rows_per_shard"] = rows_per_shard
                except Exception as e:
                    self.logger.warning(f"Could not read shard {first_shard}: {e}")
                    config_info["num_rows"] = len(config_info["shards"]) * 10000  # Estimate

        return configs_data

    @staticmethod
    async def next_pages(
        offset: int, n_pages: int, seed: str, num_rows_per_page: int = 100,
        r2_account_id: str = None,
        r2_bucket_name: str = None,
        r2_access_key_id: str = None,
        r2_secret_access_key: str = None,
    ):
        """
        Deterministically select pages based on offset and seed.

        Uses the same seeding mechanism as DatasetLoader for consistency:
        - seed: typically uid + local_rank
        - offset: blockchain block number

        This ensures the same UID gets the same data pages for a given block.
        """
        # Create a temporary instance to fetch configs
        loader = R2DatasetLoader(
            batch_size=1,
            sequence_length=1024,
            tokenizer=None,  # Not needed for page selection
            r2_account_id=r2_account_id or os.getenv("R2_DATASET_ACCOUNT_ID"),
            r2_bucket_name=r2_bucket_name or os.getenv("R2_DATASET_BUCKET_NAME"),
            r2_access_key_id=r2_access_key_id or os.getenv("R2_DATASET_ACCESS_KEY_ID"),
            r2_secret_access_key=r2_secret_access_key or os.getenv("R2_DATASET_SECRET_ACCESS_KEY"),
        )

        configs_data = await loader.fetch_dataset_configs()
        keys = sorted(configs_data.keys())

        # Use the same RNG seeding as DatasetLoader
        rng = np.random.default_rng(hash(seed) & 0xFFFFFFFF)
        rng.bit_generator.advance(offset)

        result = []
        for _ in range(n_pages):
            idx = rng.integers(0, len(keys))
            cfg = keys[idx]
            config = rng.choice(list(configs_data.keys()))
            max_row = configs_data[cfg]["num_rows"] - 1 - num_rows_per_page
            if max_row < 0:
                max_row = 0
            choice = rng.integers(0, max(1, max_row))
            result.append((cfg, int(choice), configs_data[cfg].get("split", "train")))

        return result

    async def _fetch(self, page_info: typing.Tuple[str, int, str], batch_size: int = 5):
        """Fetch data for specified pages."""
        self.pages = list(page_info)

        for i in range(0, len(self.pages), batch_size):
            batch = self.pages[i : i + batch_size]
            tasks = [
                self._fetch_data_for_page((config_name, page, split))
                for (config_name, page, split) in batch
            ]
            await asyncio.gather(*tasks)

    async def _fetch_data_to_buffer(self, num_pages):
        """Randomly sample pages and add their data to the buffer."""
        self.pages = []
        pages_to_fetch = self.get_random_pages(num_pages)

        tasks = [self._fetch_data_for_page(page) for page in pages_to_fetch]
        await asyncio.gather(*tasks)

    async def _fetch_data_for_page(self, page):
        """
        Fetch data for a single page from R2 parquet files.

        Args:
            page: Tuple of (config_name, row_offset, split)
        """
        retry_limit = self.retry_limit
        attempt = 0

        while attempt < retry_limit:
            config_name, row_offset, split = page

            try:
                # Fetch rows from parquet file(s)
                rows = await asyncio.to_thread(
                    self._fetch_rows_from_parquet_sync,
                    config_name,
                    row_offset,
                    self.num_rows_per_page,
                )

                # Tokenize the content
                buffer_to_append = []
                for text in rows:
                    if self.tokenizer is not None:
                        input_ids = await asyncio.to_thread(
                            self.tokenizer.encode,
                            text,
                            truncation=True,
                            max_length=self.sequence_length,
                        )
                        input_ids.append(self.tokenizer.eos_token_id)
                        buffer_to_append.extend(input_ids)

                async with self.lock:
                    self.buffer.extend(buffer_to_append)
                    self.pages.append((config_name, row_offset, split))
                break

            except Exception as e:
                attempt += 1
                if attempt < retry_limit:
                    self.logger.debug(
                        f"Retrying page {page} due to error: {e}. Attempt {attempt}"
                    )
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    raise Exception(
                        f"Maximum retry attempts exceeded for page {page}"
                    ) from e

    def _fetch_rows_from_parquet_sync(
        self, config_name: str, row_offset: int, num_rows: int
    ) -> typing.List[str]:
        """
        Synchronously fetch rows from parquet files in R2.

        The parquet files can be organized in two ways:
        1. Single file per config: {config_name}.parquet
        2. Sharded files: {config_name}/{shard_id}.parquet

        Args:
            config_name: Name of the config/subset
            row_offset: Starting row number
            num_rows: Number of rows to fetch

        Returns:
            List of text strings
        """
        s3 = self._get_s3_client()

        config_info = self.configs_data.get(config_name, {})
        shards = config_info.get("shards", [])
        rows_per_shard = config_info.get("rows_per_shard", 10000)

        if shards:
            # Sharded structure - find the right shard
            shard_idx = row_offset // rows_per_shard
            local_offset = row_offset % rows_per_shard

            if shard_idx >= len(shards):
                shard_idx = len(shards) - 1
                local_offset = 0

            shard_key = shards[shard_idx]
        else:
            # Try single file structure
            shard_key = f"{config_name}.parquet"
            local_offset = row_offset

        try:
            response = s3.get_object(Bucket=self.r2_bucket_name, Key=shard_key)
            parquet_data = response["Body"].read()

            # Read parquet file
            table = pq.read_table(io.BytesIO(parquet_data))
            df = table.to_pandas()

            # Extract rows
            end_offset = min(local_offset + num_rows, len(df))
            if local_offset >= len(df):
                local_offset = max(0, len(df) - num_rows)
                end_offset = len(df)

            # Get text column (try common column names)
            text_column = None
            for col_name in ["text", "content", "data", "raw_content"]:
                if col_name in df.columns:
                    text_column = col_name
                    break

            if text_column is None:
                # Use first string column
                for col in df.columns:
                    if df[col].dtype == object:
                        text_column = col
                        break

            if text_column is None:
                raise ValueError(f"No text column found in parquet file {shard_key}")

            rows = df[text_column].iloc[local_offset:end_offset].tolist()
            return [str(row) for row in rows if row is not None]

        except Exception as e:
            self.logger.error(f"Error fetching from {shard_key}: {e}")
            raise

    def get_random_pages(self, num_pages):
        """Randomly sample pages."""
        pages = []

        for _ in range(num_pages):
            config_name = random.choice(list(self.configs_data.keys()))
            config_info = self.configs_data[config_name]
            max_page = config_info["num_rows"] - 1 - self.num_rows_per_page
            if max_page < 0:
                max_page = 0
            page = random.randint(0, max(0, max_page))
            split = config_info.get("split", "train")
            pages.append((config_name, page, split))

        return pages

    def get_page_names(self):
        """Return page names as strings."""
        page_names = []
        if hasattr(self, "pages"):
            page_names = [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]
        return page_names


def get_dataset_loader(data_source: str = "huggingface"):
    """
    Factory function to get the appropriate DatasetLoader class based on config.

    Args:
        data_source: Either "huggingface" or "r2"

    Returns:
        The appropriate DatasetLoader class (not an instance)

    Usage:
        DataLoader = get_dataset_loader(config.neuron.data_source)
        pages = await DataLoader.next_pages(offset=block, n_pages=n_pages, seed=seed)
        dataset = await DataLoader.create(...)
    """
    if data_source == "r2":
        return R2DatasetLoader
    else:
        return DatasetLoader


async def create_dataset(
    data_source: str = "huggingface",
    batch_size: int = None,
    sequence_length: int = None,
    num_pages: int = None,
    pages_info=None,
    tokenizer=None,
    pack_samples: bool = False,
):
    """
    Convenience function to create a dataset using the appropriate loader.

    Args:
        data_source: Either "huggingface" or "r2"
        batch_size: Batch size for training
        sequence_length: Sequence length for tokenization
        num_pages: Number of pages to fetch (if pages_info not provided)
        pages_info: Pre-selected pages to fetch
        tokenizer: Tokenizer to use
        pack_samples: Whether to pack samples

    Returns:
        An instance of either DatasetLoader or R2DatasetLoader
    """
    LoaderClass = get_dataset_loader(data_source)
    return await LoaderClass.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_pages=num_pages,
        pages_info=pages_info,
        tokenizer=tokenizer,
        pack_samples=pack_samples,
    )


async def get_next_pages(
    data_source: str = "huggingface",
    offset: int = 0,
    n_pages: int = 1,
    seed: str = "",
    num_rows_per_page: int = 100,
):
    """
    Get next pages using the appropriate loader.

    Args:
        data_source: Either "huggingface" or "r2"
        offset: Block number offset for RNG
        n_pages: Number of pages to select
        seed: Seed for RNG (typically uid + local_rank)
        num_rows_per_page: Rows per page

    Returns:
        List of (config_name, row_offset, split) tuples
    """
    LoaderClass = get_dataset_loader(data_source)
    return await LoaderClass.next_pages(
        offset=offset,
        n_pages=n_pages,
        seed=seed,
        num_rows_per_page=num_rows_per_page,
    )
