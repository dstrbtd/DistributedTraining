import os
import json
import yaml
import time
import asyncio
import torch
import s3fs
import pyarrow.parquet as pq
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer
import bittensor as bt
import random
import hashlib


class BatchLoader:
    def __init__(self, tokenizer=None, batch_size=None, sequence_length=None):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.buffer = []
        self._data = None
        self._num_batches = 0
        self._batch_idx = 0

    def reduce_buffer_size(self, target_size: int = None, fraction: float = None, method: str = "sample"):
        """
        Reduce the buffer size by either absolute count or fraction.
        
        Args:
            target_size: Desired number of tokens in buffer
            fraction: Fraction of buffer to keep (e.g., 0.2 for 20%)
            method: "truncate" or "sample" (random sampling)
        """
        if target_size is None and fraction is None:
            return
            
        if fraction is not None:
            target_size = int(len(self.buffer) * fraction)
        
        original_size = len(self.buffer)  # Store original size for debug message
        
        if len(self.buffer) <= target_size:
            return
        
        if method == "truncate":
            self.buffer = self.buffer[:target_size]
        elif method == "sample":
            rng = self.generate_rng("buffer_reduction")
            self.buffer = rng.sample(self.buffer, target_size)
        
        if self.debug:
            self.logger.debug(f"Buffer reduced from {original_size} to {target_size} tokens ({method})")

    def prepare_batches(self, batch_size=None, sequence_length=None, device="cpu"):
        batch_size = batch_size or self.batch_size
        sequence_length = sequence_length or self.sequence_length

        token_buffer = self.buffer
        total_tokens = len(token_buffer)
        num_sequences = total_tokens // sequence_length
        trimmed_tokens = token_buffer[: num_sequences * sequence_length]

        data = torch.tensor(trimmed_tokens, dtype=torch.long, device=device)
        data = data.view(num_sequences, sequence_length)
        num_batches = num_sequences // batch_size

        self._data = data
        self._num_batches = num_batches
        self._batch_idx = 0
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # if self.debug and num_batches > 0:
        #     first_batch = data[:batch_size]
        #     self.logger.debug("Preview tokens:", first_batch[0][:5].tolist())

    def __iter__(self):
        if self._data is None:
            raise RuntimeError("Call prepare_batches() before iterating.")
        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self._num_batches:
            raise StopIteration
        i = self._batch_idx
        batch = self._data[i * self.batch_size : (i + 1) * self.batch_size]
        self._batch_idx += 1
        return batch, batch.clone()

    def __len__(self):
        return self._num_batches

class DatasetLoader(BatchLoader):
    def __init__(
        self,
        uid: int,
        current_block: int = 0,
        tokenizer=None,

        max_configs: int = 3,
        max_shards: int = 3,
        max_row_groups: int = 4,
        max_rows_per_group: int = 100,

        batch_size: int = 4,
        sequence_length: int = 1024,

        debug: bool = True,
        randomness: bool = True,
    ):
        super().__init__(
            tokenizer=tokenizer, 
            batch_size=batch_size, 
            sequence_length=sequence_length
        )

        self.uid = uid
        self.current_block = current_block
        self.logger = bt.logging
        load_dotenv(find_dotenv())

        self.max_configs = max_configs
        self.max_shards = max_shards
        self.max_row_groups = max_row_groups
        self.max_rows_per_group = max_rows_per_group
        
        self.debug = debug
        self.randomness = randomness
        # self.debug and self.logger.debug(f"self.max_configs: {self.max_configs}, self.max_shards: {self.max_shards}, self.max_row_groups: {self.max_row_groups}, self.max_rows_per_group: {self.max_rows_per_group}")

        def require_env(name: str) -> str:
            val = os.getenv(name)
            if not val:
                raise ValueError(f"{name} env var not set")
            return val

        self.BUCKET      = require_env(f"R2_BUCKET_NAME")
        self.ACCOUNT_ID  = require_env(f"R2_ACCOUNT_ID")
        self.ACCESS_KEY  = require_env(f"R2_ADMIN_ACCESS_KEY_ID")
        self.SECRET_KEY  = require_env(f"R2_ADMIN_SECRET_ACCESS_KEY")
        # self.logger.debug(f"self.BUCKET: {self.BUCKET}")
        # self.logger.debug(f"self.ACCOUNT_ID: {self.ACCOUNT_ID}")
        # self.logger.debug(f"self.ACCESS_KEY: {self.ACCESS_KEY}")
        # self.logger.debug(f"self.SECRET_KEY: {self.SECRET_KEY}")

        self.DATASET = "HuggingFaceFW_fineweb-edu-score-2"
        self.META_NAME = "_metadata.yaml"
        self.SHARD_NAME = "_shard_sizes.json"

        self.CACHE_DIR = ".cache"
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.meta_cache_path = os.path.join(self.CACHE_DIR, self.META_NAME)
        self.shard_cache_path = os.path.join(self.CACHE_DIR, self.SHARD_NAME)

        self.fs = s3fs.S3FileSystem(
            key=self.ACCESS_KEY,
            secret=self.SECRET_KEY,
            client_kwargs={"endpoint_url": f"https://{self.ACCOUNT_ID}.r2.cloudflarestorage.com"},
        )

        self.metadata = {}
        self.shard_sizes = {}

        self.total_row_groups_loaded = 0
        self.total_rows_loaded = 0

        self.debug and self.logger.debug(f"DatasetLoader initialized with UID={self.uid}, block={self.current_block}")

    def generate_rng(self, context: str = "") -> random.Random:
        """
        Returns a reproducible RNG based on the stored UID and current block.
        """
        seed_str = f"{self.uid}-{context}-{self.current_block}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        return random.Random(seed)
    
    def select_configs(self, configs):
        rng = self.generate_rng("config_selection")
        n = min(len(configs), self.max_configs)
        indexes = rng.sample(range(len(configs)), n)
        self.debug and self.logger.debug(f"Config idxs chosen: {indexes}")
        return [configs[i] for i in indexes]

    def select_shards(self, shards, context="shard_selection"):
        rng = self.generate_rng(context)
        n = min(len(shards), self.max_shards)
        indexes = rng.sample(range(len(shards)), n)
        self.debug and self.logger.debug(f"Shard idxs chosen: {indexes}")
        return [shards[i] for i in indexes]

    def select_row_groups(self, num_row_groups, context="row_group"):
        rng = self.generate_rng(context)
        start_idx = rng.randint(0, num_row_groups - self.max_row_groups) if num_row_groups > self.max_row_groups else 0
        rg_indices = list(range(start_idx, start_idx + self.max_row_groups))
        return rg_indices

    def select_rows(self, num_rows, context="row"):
        rng = self.generate_rng(context)
        start_idx = rng.randint(0, num_rows - self.max_rows_per_group) if num_rows > self.max_rows_per_group else 0
        end_idx = min(start_idx + self.max_rows_per_group, num_rows)
        return start_idx, end_idx

    async def load_bucket_data_to_buffer(self):
        """Load data from bucket into buffer."""
        
        if not self.metadata or not self.shard_sizes:
            self.load_bucket_configs()

        all_shards = await self.get_shards_from_configs()
        start_time = time.perf_counter()

        self.buffer = await self.fetch_data_for_shards(
            shard_paths=all_shards, 
        )

        end_time = time.perf_counter()
        if self.debug:
            self.logger.debug(f"Buffer length: {len(self.buffer)}")
            self.logger.debug(f"load_bucket_data_to_buffer took {end_time - start_time:.2f}s\n")

        return self.buffer

    def load_bucket_configs(self):
        self.download_config(f"{self.BUCKET}/{self.DATASET}/{self.META_NAME}", self.meta_cache_path)
        self.download_config(f"{self.BUCKET}/{self.DATASET}/{self.SHARD_NAME}", self.shard_cache_path)

        with open(self.meta_cache_path, "r") as f:
            self.metadata = yaml.safe_load(f)

        with open(self.shard_cache_path, "r") as f:
            self.shard_sizes = json.load(f)

    def download_config(self, remote_path, local_path):
        if os.path.exists(local_path):
            return
        data = self.fs.cat(remote_path)
        with open(local_path, "wb") as dst:
            dst.write(data)            

    async def get_shards_from_configs(self):
        configs = await self.get_configs()
        configs = self.select_configs(configs)

        shard_lists = await asyncio.gather(
            *(asyncio.to_thread(self.list_shard_files, c) for c in configs)
        )

        all_shards = []
        for shards in shard_lists:
            selected = self.select_shards(shards, context=f"shard_{shards[0] if shards else ''}")
            all_shards.extend(selected)

        # self.debug and self.logger.debug(f"All_shards: {all_shards}\n")
        return all_shards          

    async def get_configs(self):
        all_configs = [c.get("config_name") for c in self.metadata.get("configs", []) if c.get("config_name")]
        async def check_config(config):
            config_path = f"{self.BUCKET}/{self.DATASET}/{config}"
            exists = await asyncio.to_thread(self.fs.exists, config_path)
            return config if exists else None
        results = await asyncio.gather(*(check_config(c) for c in all_configs))
        return [r for r in results if r]

    def list_shard_files(self, config):
        config_info = self.shard_sizes.get(config, {})
        shards = config_info.get("shards", [])
        return [shard["path"] for shard in shards]
            
    async def fetch_data_for_shards(self, shard_paths):
        semaphore = asyncio.Semaphore(10)
        async def load_with_limit(shard):
            async with semaphore:
                return await self.load_shard(shard_path=shard)
        results = await asyncio.gather(*(load_with_limit(p) for p in shard_paths))
        return [token for shard_buffer in results for token in shard_buffer]
    
    async def load_shard(self, shard_path):
        buffer = []
        try:
            reader = await asyncio.to_thread(pq.ParquetFile, f"s3://{shard_path}", filesystem=self.fs)
        except Exception as e:
            self.logger.debug(f"Failed to open shard {shard_path}: {e}")
            return buffer

        num_row_groups = reader.num_row_groups
        rg_indices = self.select_row_groups(num_row_groups, context=f"row_group_{shard_path}")

        for rg_idx in rg_indices:
            row_group = await asyncio.to_thread(reader.read_row_group, rg_idx, columns=["text"], use_threads=True)
            num_rows = len(row_group)
            start_idx, end_idx = self.select_rows(num_rows, context=f"row_{shard_path}_rg{rg_idx}")
            rows = row_group.slice(offset=start_idx, length=end_idx - start_idx)

            encodings = await self.tokenize_texts(rows["text"].to_pylist())
            for ids in encodings:
                ids.append(self.tokenizer.eos_token_id)
                buffer.extend(ids)

            self.total_row_groups_loaded += 1
            self.total_rows_loaded += len(rows)

        return buffer
    
    async def tokenize_texts(self, texts):
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda t=text: self.tokenizer.encode(
                    t,
                    truncation=True,
                    max_length=self.sequence_length
                )
            )
            for text in texts
        ]
        encoded = await asyncio.gather(*tasks)
        return encoded

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("dstrbtd/llama-1b", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    miner_uid = 18
    current_block = 5962593    

    loader = DatasetLoader(
        tokenizer=tokenizer,
        uid=miner_uid,
        current_block=current_block,
        max_configs=1,
        # max_rows_per_group=100,
        # sequence_length=1024,
        # batch_size=4,
        # debug=True,
        # randomness=True
    )

    asyncio.run(loader.load_bucket_data_to_buffer())

    loader.reduce_buffer_size(
        fraction=0.3,
        method="truncate",
    )
    
    loader.prepare_batches()    

    for i, (inputs, labels) in enumerate(loader):
        print(f"Batch {i}: input_ids shape {inputs.shape}")
        print(f"Batch {i}: labels shape {labels.shape}")
        if i >= 1:
            break