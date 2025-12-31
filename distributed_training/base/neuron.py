# The MIT License (MIT)
# Copyright © 2025 dstrbtd.ai

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

import time
import os
import pathlib
import copy
import boto3
import threading
from abc import ABC, abstractmethod

import bittensor as bt

from distributed_training import __spec_version__ as spec_version
from botocore.config import Config

# Sync calls set weights and also resyncs the metagraph.
from distributed_training.utils.config import (
    add_args,
    check_config,
    config,
    R2Access,
    R2Config,
)
# from distributed_training.utils.logger import setup_logging
from distributed_training.utils.misc import ttl_get_block
from dotenv import load_dotenv


import torch, torch.distributed as dist
import datetime as dt

load_dotenv()


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        self.current_block = ttl_get_block(self)
        return self.current_block

    def set_current_block_across_ranks(self):
        current_block_tensor = (
            torch.tensor([self.current_block]) if self.master else torch.tensor([0])
        )
        dist.broadcast(current_block_tensor, src=0, group=self.gloo_group)
        self.current_block = current_block_tensor[0].item()

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Set up logging with the provided configuration and directory.
        bt.logging.set_config(config=self.config.logging)
        self.logger = bt.logging

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        self.logger.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        self.logger.info("Setting up bittensor objects.")

        # Set distributed variables
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.master = self.local_rank == 0

        if self.master:
            # The wallet holds the cryptographic key pairs for the miner.
            self.wallet = bt.wallet(config=self.config)
            self.logger.info(f"Wallet: {self.wallet}")

        if not dist.is_initialized():
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="tcp://127.0.0.1:29500",
                    rank=self.local_rank,
                    world_size=self.world_size,
                    # timeout=dt.timedelta(seconds=1800),
                )
            if not hasattr(self, "gloo_group"):
                self.gloo_group = dist.new_group(
                    backend="gloo",
                )

        if self.master:
            # The subtensor is our connection to the Bittensor blockchain.
            self.subtensor = bt.subtensor(config=self.config)
            self.logger.info(f"Subtensor: {self.subtensor}")

            # The metagraph holds the state of the network, letting us know about other validators and miners.
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.logger.info(f"Metagraph: {self.metagraph}")

            # Check if the miner is registered on the Bittensor network before proceeding further.
            self.check_registered()

            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            self.logger.info(
                f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
            )
        else:
            self.uid = 0

        uid = torch.tensor([self.uid], device="cpu")
        dist.barrier(group=self.gloo_group)
        dist.broadcast(uid, src=0, group=self.gloo_group)
        dist.barrier(group=self.gloo_group)
        self.uid = uid[0].item()

        master_uid = (
            torch.tensor(
                [
                    self.metagraph.hotkeys.index(
                        self.config.neuron.master_ss58_address,
                    )
                ]
            )
            if self.master
            else torch.tensor([0])
        )
        dist.broadcast(master_uid, src=0, group=self.gloo_group)
        self.master_uid = master_uid[0].item()

        # Setup Logging
        # setup_logging(self, config=self.config)

        # Create the R2 data model
        r2 = R2Config(
            bucket_name=f"{self.config.neuron.global_model_name.split('/')[-1]}-{self.uid:03d}"
            if "miner" in self.__class__.__name__.lower()
            else self.config.neuron.global_model_name,
            account_id=os.getenv("R2_ACCOUNT_ID"),
            read=R2Access(
                access_key_id=os.getenv("R2_READ_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("R2_READ_SECRET_ACCESS_KEY"),
            ),
            write=R2Access(
                access_key_id=os.getenv("R2_WRITE_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("R2_WRITE_SECRET_ACCESS_KEY"),
            ),
        )
        self.config.r2 = r2

        # Save directory
        self.output_dir = os.path.join(os.getcwd(), self.config.r2.bucket_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(
            os.path.join(os.getcwd(), self.config.neuron.global_model_name),
            exist_ok=True,
        )

        # Init Step
        self.step = 0

        # Initialize the all_reduce, download and upload variables.
        self.allreduce_timeout = 840
        self.upload_state_duration = 1800
        self.all_reduce_success_status = True
        self.should_all_reduce = False
        self.retry_limit = 100
        self.retry_delay = 60

        # Create different r2 sessions
        r2_config = Config(
            retries={"max_attempts": 10, "mode": "adaptive"},  # or "standard"
            connect_timeout=30,
            read_timeout=120,
            max_pool_connections=50,
        )
        self.session = boto3.session.Session()
        self.r2 = {
            "local": self.session.client(
                "s3",
                endpoint_url=f"https://{self.config.r2.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=self.config.r2.read.access_key_id,
                aws_secret_access_key=self.config.r2.read.secret_access_key,
                region_name="auto",
                config=r2_config,
            )
        }
        self.r2["write"] = boto3.client(
            "s3",
            endpoint_url=f"https://{self.config.r2.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=self.config.r2.write.access_key_id,
            aws_secret_access_key=self.config.r2.write.secret_access_key,
            region_name="auto",
            config=r2_config,
        )
        commitment = None
        while commitment == None:
            try:
                if self.master:
                    commitment = [
                        self.subtensor.get_commitment(
                            self.config.netuid, self.master_uid
                        )
                    ]
                else:
                    commitment = [
                        self.config.r2.account_id
                        + self.config.r2.read.access_key_id
                        + self.config.r2.read.secret_access_key
                    ]
                dist.broadcast_object_list(commitment, src=0, group=self.gloo_group)
                global_account_id = commitment[0][:32]
                global_access_key_id = commitment[0][32:64]
                global_asecret_access_key = commitment[0][64:]
                self.r2["global"] = self.session.client(
                    "s3",
                    endpoint_url=f"https://{global_account_id}.r2.cloudflarestorage.com",
                    aws_access_key_id=global_access_key_id,
                    aws_secret_access_key=global_asecret_access_key,
                    region_name="auto",
                    config=r2_config,
                )
            except Exception as e:
                self.logger.info(f"Error getting commitment: {str(e)}")
                time.sleep(15)

        self.reload_state_event = threading.Event()

    # @abstractmethod # miner is not using this anymore
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        ...

    @abstractmethod
    def run(self):
        ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        if self.master:
            try:
                # Ensure miner or validator hotkey is still registered on the network.
                self.check_registered()

                if self.should_sync_metagraph():
                    self.resync_metagraph()

                if self.should_set_weights():
                    self.logger.info("Should Set Weights")
                    self.set_weights()

                if self.should_sync_metagraph():
                    self.metagraph.last_update[self.uid] = self.block

                if (self.step != 0) and (self.neuron_type != "MinerNeuron"):
                    self.save_state()
            except Exception as e:
                self.logger.debug("Sync failed with error {e}")

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            self.logger.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "MinerNeuron"  # don't set weights if you're a miner

    def save_state(self):
        self.logger.warning(
            "save_state() not implemented for this neuron. You can implement this function to save model checkpoints or other useful data."
        )

    def load_state(self):
        self.logger.warning(
            "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        )

    def print_memory_usage(self):
        def cg_read(p):
            try:
                return pathlib.Path(p).read_text().strip()
            except FileNotFoundError:
                return None

        memory_used = 0
        memory_limit = 0
        memory_used_gb = 0
        memory_limit_gb = 0

        # Memory limit (bytes) — cgroup v2 then v1
        memory_limit = cg_read("/sys/fs/cgroup/memory.max") or cg_read(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )
        if memory_limit and memory_limit != "max":
            memory_limit_gb = int(memory_limit) / 1024**3
            self.logger.debug(f"Memory limit: {memory_limit_gb:.1f} GB")
        else:
            self.logger.debug("Memory limit: Unlimited Or Not Set")

        memory_used = cg_read("/sys/fs/cgroup/memory.current") or cg_read(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        )
        if memory_used and memory_used != "max":
            memory_used_gb = int(memory_used) / 1024**3
            self.logger.debug(f"Memory Used: {memory_used_gb:.1f} GB")
        else:
            self.logger.debug("Memory Used: Unlimited Or Not Set")

        if self.master:
            self.logger.debug(
                f"CPU Memory Usage: {memory_used_gb:.1f}GBs out of {memory_limit_gb:.1f}GBs"
            )

        return (
            f"CPU Memory Usage: {memory_used_gb:.1f}GBs out of {memory_limit_gb:.1f}GBs"
        )
