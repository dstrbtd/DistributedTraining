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
import random
import typing

import aiohttp
import numpy as np
import torch
import time
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


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
