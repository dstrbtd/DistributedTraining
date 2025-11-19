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


import boto3
import os
import time

os.environ["NEST_ASYNCIO"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Set seed and enable deterministic settings to ensure reproducibility
import numpy as np
import random
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import math
import threading
import torch.distributed as dist

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from transformers import AutoTokenizer

from distributed_training.base.validator import BaseValidatorNeuron
from distributed_training.utils.chain import log_r2_to_chain
from distributed_training.utils.r2 import log_peerid_to_r2
from distributed_training.utils.misc import (
    init_dht,
    load_wandb,
)
from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    get_progress,
)
from random import randrange
from distributed_training.utils.state_loader import (
    cleanup_old_cache,
    load_state_from_peer,
)
from distributed_training.utils.uids import map_uid_to_peerid, update_run_peerid_list
from distributed_training.validator import forward
from distributed_training import __run__

from distributed_training.utils.compression import CompressDCT, TransformDCT
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from distributed_training.utils.r2 import r2_download
from distributed_training.utils.state_loader import get_r2_client


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.set_current_block_across_ranks()
        self._update_wandb_project()
        self._init_basic_components()
        self._init_model_components()
        self._init_network_components()
        self._init_uid_components()
        self._load_gradient_compressors()

    def test_gradient(self, epoch, uid):
        try:
            prefix = f"epoch-{epoch}/"
            destination_dir = os.path.join(os.getcwd(), "gradients", prefix)
            final_name = f"uid-{uid:03d}-epoch-{epoch}.pt"
            final_path = os.path.join(destination_dir, final_name)

            r2 = get_r2_client(self, uid, donwload_on_all_ranks=True)
            gradient_path = r2_download(
                self,
                r2=r2,
                bucket=f"{self.config.neuron.global_model_name.split('/')[-1]}-{uid:03d}",
                key=f"{prefix}gradients.pt",
                donwload_on_all_ranks=False,
                run_on_all_ranks=False,
                destination=destination_dir,
            )
            if self.master:
                os.makedirs(destination_dir, exist_ok=True)
                # if r2_download returned the directory, assume file is "<dest_dir>/gradients.pt"
                src = (
                    gradient_path
                    if gradient_path.endswith(".pt")
                    else os.path.join(destination_dir, "gradients.pt")
                )
                if os.path.abspath(src) != os.path.abspath(final_path):
                    os.replace(src, final_path)  # atomic on POSIX

            gradient_path = final_path
        except Exception as e:
            self.logger.info(f"Failed to test gradients with error {e}")
        finally:
            dist.barrier()

    def _update_wandb_project(self):
        suffix = "_validators" if self.neuron_type == "ValidatorNeuron" else "_miners"
        self.config.neuron.wandb_project += suffix

    def report_allreduce_scores(
        self,
        op_id,
        epoch,
        validator_uid,
        success_rate,
        duration,
        participating_miners_count,
        failed_miners_count,
        bandwidth=None,
    ):
        """Report AllReduce operation metrics to InfluxDB"""
        try:
            point = (
                Point("allreduce_operations")
                .tag("operation_id", str(op_id))
                .tag("epoch", str(epoch))
                .tag("run_id", __run__)
                .tag("validator_uid", str(validator_uid))
                .tag("run_id", __run__)
                .field("success_rate", float(success_rate))
                .field("duration", float(duration))
                .field(
                    "learning_rate", float(self.inner_optimizer.param_groups[0]["lr"])
                )
                .field("participating_miners", int(participating_miners_count))
                .field("failed_miners", int(failed_miners_count))
            )

            if bandwidth is not None:
                point = point.field("bandwidth", float(bandwidth))

            self.influx_write_api.write(
                bucket=self.config.neuron.influxdb_bucket,
                org=self.config.neuron.influxdb_org,
                record=point,
            )
            self.logger.info(
                f"Validator {validator_uid} reported AllReduce operation {op_id} metrics to InfluxDB"
            )
        except Exception as e:
            self.logger.error(f"Error reporting AllReduce metrics: {e}")

    def report_train_scores(self):
        """Send validator scoring metrics to InfluxDB"""
        try:
            points = []

            for uid, data in self.uid_tracker.items():
                fields = self.flatten_model(data)  # dotted key dict
                point = (
                    Point("miner_scores")
                    .tag("validator_uid", str(self.uid))
                    .tag("validator_hotkey", self.wallet.hotkey.ss58_address)
                    .tag("miner_uid", str(uid))
                    .tag("run_id", __run__)
                )
                for k, v in fields.items():
                    if isinstance(
                        v, (int, float, bool)
                    ):  # only store valid Influx types
                        if (
                            (k == "all_reduce.count")
                            or (k == "chaindata.last_updated_block")
                            or (k == "uid")
                        ):
                            point = point.field(k, v)
                        else:
                            point = point.field(k, float(v))
                    elif (k == "all_reduce.peer_id") or (k == "train.model_id"):
                        point = point.field(k, v)

                points.append(point)

                if uid in self.openskill_ratings.keys():
                    point = (
                        Point("openskill_scores")
                        .tag("validator_uid", str(self.uid))
                        .tag("validator_hotkey", self.wallet.hotkey.ss58_address)
                        .tag("miner_uid", str(uid))
                        .tag("run_id", __run__)
                        .field("mu", float(self.openskill_ratings[uid].mu))
                        .field("sigma", float(self.openskill_ratings[uid].sigma))
                        .field("ordinal", float(self.openskill_ratings[uid].ordinal()))
                    )
                    points.append(point)

            # Write points to InfluxDB
            self.influx_write_api.write(
                bucket=self.config.neuron.influxdb_bucket,
                org=self.config.neuron.influxdb_org,
                record=points,
            )
        except Exception as e:
            self.logger.error(f"Error reporting scoring metrics: {e}")

    def _init_metrics_collection(self):
        if self.master:
            # Initialize InfluxDB client
            self.influx_client = None
            self.influx_write_api = None
            try:
                self.logger.info(
                    "Attempting to initialize InfluxDB client for metrics collection..."
                )
                self.influx_client = InfluxDBClient(
                    url=self.config.neuron.influxdb_url,
                    token=self.config.neuron.influxdb_token,
                    org=self.config.neuron.influxdb_org,
                )
                self.influx_write_api = self.influx_client.write_api(
                    write_options=SYNCHRONOUS
                )
                self.logger.info(
                    "InfluxDB client and write_api initialized successfully."
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to initialize InfluxDB client: {e}. Metrics collection will be disabled."
                )
                if self.influx_client:
                    try:
                        self.influx_client.close()
                    except Exception as close_e:
                        self.logger.error(
                            f"Error closing InfluxDB client during cleanup: {close_e}"
                        )
                self.influx_client = None
                self.influx_write_api = None

    def _init_basic_components(self):
        """Initialize basic validator components"""
        # Core setup
        self.device = self.config.neuron.device
        init_dht(self)

        # Progress tracking
        self._init_progress_tracking()

        # Wandb setup
        if (not self.config.neuron.dont_wandb_log) and self.master:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "validator", str(self.dht.peer_id)
            )

        # Tracking metrics
        self._init_metrics_collection()

    def _init_progress_tracking(self):
        self.local_progress = LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes() if self.master else b"",
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
            inner_step=0,
            loss=0.0,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        self.global_progress.epoch = get_progress(self, "global")[0]
        self.local_progress.epoch = self.global_progress.epoch

        if self.global_progress.epoch is None:
            self.logger.error(
                "Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )

    def _init_model_components(self):
        """Initialize model-related components including tokenizer and optimizer settings."""
        self._setup_model_params()
        self._init_tokenizer()
        self._setup_model_state()
        self._setup_training_params()
        self._setup_uid_api()

    def _setup_model_params(self):
        # Timeouts
        self.load_state_timeout = 180

        # Core parameters
        self.learning_rate_maximum = 2.5e-4
        self.learning_rate_eval = 0.5
        self.weight_decay = 0.1
        self.num_inner_steps = 500
        self.offload_optimizer = True
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 10

        # Validator-specific training parameters
        self.maximum_steps = 306 * 4  # 10_000_000_000/(32000*1024)
        self.warmup_steps = 62  # 306 / 5
        self.failed_is_alive_counter_threshold = 10

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.neuron.global_tokenizer_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_model_state(self):
        self.learning_rate = self.get_learning_rate()
        self.average_loss = None

        load_state_from_peer(self, self.master_uid, self.global_progress.epoch)
        cleanup_old_cache(self)

        if self.local_progress.epoch < self.global_progress.epoch:
            load_state_from_peer(self, epoch=self.global_progress.epoch)

    def _setup_training_params(self):
        self.local_batch_size_train = self.config.neuron.local_batch_size_train
        self.local_batch_size_train_effective = (
            self.config.neuron.local_batch_size_train_effective / self.world_size
        )
        self.logging_interval = 5
        self.number_of_local_steps = (
            self.config.neuron.local_batch_size_train_effective
            // self.config.neuron.local_batch_size_train
        )

        self.running_loss = 0.0
        self.batch_count = 0

    def _init_network_components(self):
        """Initialize network and P2P components"""
        self.logger.info("Logging PeerID to chain")
        log_r2_to_chain(self)
        log_peerid_to_r2(self)

    def _init_uid_components(self):
        self._setup_uids()
        self._init_peer_mapping()
        self._setup_allreduce_block()

    def _setup_uids(self):
        if self.master:
            self.failed_is_alive_counter = {
                uid: 0 for uid in self.metagraph.uids.tolist()
            }
            self.miner_uids = []

    def _load_gradient_compressors(self):
        self.logger.info("Load Start")
        dist.barrier(device_ids=[self.local_rank])
        opts = StateDictOptions(
            full_state_dict=True, cpu_offload=True
        )  # gather to CPU on rank 0
        full_state = get_model_state_dict(self.model, options=opts)
        self.logger.info("Full state")
        if self.master:
            # Init compression
            self.transformer = TransformDCT(
                full_state, target_chunk=self.config.neuron.target_chunk
            )
            self.compressor = CompressDCT(
                use_quantization=True,
                quantization_bins=self.config.neuron.quantization_bins,
                quantization_range=self.config.neuron.quantization_range,
            )
            self.logger.info("Compressor Loaded")
            self.xshapes = {}
            self.totalks = {}
            self.error_feedback = {}
            self.owned_params = set()
            for n, p in full_state.items():
                self.owned_params.add(n)
                self.error_feedback[n] = torch.zeros_like(p, device="cpu")
                _, _, xshape, totalk = self.compressor.compress(
                    self.transformer.encode(
                        torch.zeros_like(p), use_dct=self.config.neuron.use_dct
                    ),
                    self.config.neuron.topk_compression,
                )
                self.xshapes[n] = xshape
                self.totalks[n] = totalk
        dist.barrier(device_ids=[self.local_rank])

    def _init_peer_mapping(self):
        self.stop_event = threading.Event()
        if self.master:
            map_uid_to_peerid(self)
            update_run_peerid_list(self)

    def _setup_allreduce_block(self):
        if (self.uid == self.master_uid) or (
            "last_allreduce_block" not in self.model.config.__dict__
        ):
            self.last_allreduce_block = self.current_block
        else:
            self.last_allreduce_block = self.model.config.last_allreduce_block
        self.blocks_since_allreduce = self.current_block - self.last_allreduce_block

    def _setup_uid_api(self):
        self.uid_api_url = self.config.neuron.uid_api_url
        self.uid_api_get_token = self.config.neuron.uid_api_get_token
        self.uid_api_post_token = self.config.neuron.uid_api_post_token

    def update_local_tracker_state(self, rewards, responses):
        for reward, response in zip(rewards, responses[0]):
            if (reward != 0) and (response.dataset_indices is not None):
                self.local_progress.samples_accumulated += len(response.dataset_indices)
            else:
                continue

    def get_learning_rate(self):
        learning_rate_minimum = self.learning_rate_maximum * 0.1
        # 1) linear warmup for warmup_steps
        if self.global_progress.epoch < self.warmup_steps:
            return (
                self.learning_rate_maximum
                * (self.global_progress.epoch + 1)
                / self.warmup_steps
            )
        # 2) if epoch > lr_decay_iters, return learning_rate_minimum
        if self.global_progress.epoch > self.maximum_steps:
            return learning_rate_minimum
        # 3) if in between, use cosine decay down to min learning rate
        decay_ratio = (self.global_progress.epoch - self.warmup_steps) / (
            self.maximum_steps - self.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (learning_rate_minimum + coeff) * (
            self.learning_rate_maximum - learning_rate_minimum
        )

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.uid],
            "rank": self.metagraph.ranks[self.uid],
            "vtrust": self.metagraph.validator_trust[self.uid],
            "dividends": self.metagraph.dividends[self.uid],
            "emissions": self.metagraph.emission[self.uid],
        }

    async def forward(self):
        return await forward(self)

    def update_model_with_pseudo_gradient(
        self, model: torch.nn.Module, uid: int, state_dict: dict, quantize: bool = False
    ) -> None:
        model.zero_grad()

        opts_get = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        full_state = get_model_state_dict(self.model, options=opts_get)

        if self.master:
            # Apply the pseudo-gradient to the model parameters
            for n, p in full_state.items():
                idxs_key = n + "idxs"
                vals_key = n + "vals"
                quant_key = n + "quant_params"
                idxs = state_dict.get(idxs_key, None)
                vals = state_dict.get(vals_key, None)
                quant_params = state_dict.get(quant_key, None)
                if (
                    (idxs is not None)
                    and (vals is not None)
                    and ((quant_params is not None) or (quantize is False))
                ):
                    idxs = idxs.to(self.device)
                    vals = vals.to(self.device)
                    grad = self.transformer.decode(
                        self.compressor.decompress(
                            p.to(self.device),
                            idxs,
                            vals,
                            self.xshapes[n],
                            self.totalks[n],
                            quant_params,
                        ),
                        use_dct=self.config.neuron.use_dct,
                    ).to(p.device)

                    # Final safety check on the gradient itself
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        self.logger.info(
                            f"Decompressed gradient for parameter {n} contains NaN/Inf, skipping UID {uid}"
                        )
                        raise ValueError(
                            f"Invalid gradient from UID {uid}: NaN or Inf in decompressed gradient for parameter {n}"
                        )
                    p.data.sub_(
                        grad,
                        alpha=self.outer_optimizer.param_groups[0]["lr"]
                        * self.learning_rate_eval,
                    )

        opts_set = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=True,
        )
        # Push back into the model (reshard). All ranks must enter.
        set_model_state_dict(
            model=self.model, model_state_dict=full_state, options=opts_set
        )

    def flatten_model(self, model, parent_key="", sep="."):
        """Recursively flatten a Pydantic model or dict into dotted key format."""
        items = []
        if hasattr(model, "model_dump"):  # Pydantic model
            model = model.model_dump()
        for k, v in model.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_model(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    Validator().run()
