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

# Set seed and enable deterministic settings to ensure reproducibility
import boto3
import os
import numpy as np
import random
import torch

os.environ["NEST_ASYNCIO"] = "0"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import asyncio
import gc
import os
import random
import subprocess
import time
import typing
import threading

from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import bittensor as bt
import psutil
import torch
import json
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
)
from hivemind.averaging.averager import compute_schema_hash
from huggingface_hub import (
    create_repo,
    create_tag,
    delete_tag,
    list_repo_refs,
    repo_exists,
)
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed_training
from distributed_training import __run__

# from distributed_training.averaging.avg_handler import AllReduceError
from distributed_training.base.miner import BaseMinerNeuron, TrainingStatus
from distributed_training.data.dataset import DatasetLoader
from distributed_training.utils.chain import log_r2_to_chain
from distributed_training.utils.misc import (
    init_dht,
    load_wandb,
)
from distributed_training.utils.logger import setup_logging
from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    get_progress,
)
from distributed_training.utils.state_loader import (
    cleanup_old_cache,
    load_state_from_peer,
)
from distributed_training.utils.r2 import log_peerid_to_r2
from distributed_training.utils.compression import TransformDCT, CompressDCT
import torch.distributed as dist
import distributed_training


import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
)
from distributed_training.utils.dist import gloabl_dist_checkpoint
from safetensors.torch import save_file

import faulthandler, signal

faulthandler.register(signal.SIGUSR1)


def cuda_mem(logger, tag):
    torch.cuda.synchronize()
    logger.info(
        f"[{tag}] alloc={torch.cuda.memory_allocated()/1e9:.3f} GB | "
        f"reserved={torch.cuda.memory_reserved()/1e9:.3f} GB"
    )


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self._update_wandb_project()
        self._init_basic_components()
        self._init_model_components()
        self._init_network_components()

        if self.master:
            self.block
        else:
            self.current_block = 0
        self.set_current_block_across_ranks()
        self.starting_block = self.current_block

        if self.should_sync_model:
            self.start_background_upload(
                epoch=self.global_progress.epoch,
            )
        self.all_reduce_flag = 0

        self.loop = asyncio.new_event_loop()

        if self.master:
            cuda_mem(self.logger, f"Load 0")

            # Serve passes the axon information to the network + netuid we are hosting on.
            # This will auto-update if the axon port of external ip have changed.
            self.logger.info(
                f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid} and port: {self.axon.port}"
            )
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

            # Start  starts the miner's axon, making it active on the network.
            self.axon.start()
            self.logger.info(f"Miner starting at block: {self.block}")

            self.reload_state_checker_thread = threading.Thread(
                target=self.reload_state_watcher, daemon=True
            )
            self.reload_state_checker_thread.start()

    def reload_state_watcher(self):
        """Background thread on every rank; only sets a local flag on rank 0."""
        while not self.stop_event.is_set():
            try:
                self.sync()
            except Exception as e:
                self.logger.debug(f"Error {e} when trying to sync")
            if not self.all_reduce_success_status:
                wait_time = (
                    self.allreduce_timeout
                    + self.upload_state_duration
                    - time.perf_counter()
                    + self.all_reduce_start_time
                )
                if wait_time > 0:
                    self.logger.info(
                        f"Waiting {int(wait_time)} seconds until validator complete the all_reduce"
                    )
                    # Wait for the master validator to upload new global model
                    time.sleep(wait_time)
                # Check if master validator has failed to all_reduce
                self.global_progress.epoch = get_progress(self, "global")[0]
                self.reload_state_event.set()
            # elif (
            #     self.local_progress.epoch == 0
            #     and (self.local_progress.inner_step % 10 == 0)
            # ) or (
            #     self.local_progress.epoch != 0
            #     and (self.local_progress.inner_step % 2 == 0)
            # ):
            #     # ) and (self.all_reduce_flag != 1):
            #     #     # elif (datetime.datetime.now().minute % 30 == 0) and (
            #     #     #     self.all_reduce_flag != 1
            #     #     # ):
            #     self.loop.run_until_complete(
            #         self.all_reduce(
            #             distributed_training.protocol.AllReduce(
            #                 min_group_size=self.config.neuron.min_group_size,
            #                 timeout=420,
            #             )
            #         )
            #     )
            #     time.sleep(self.allreduce_timeout + self.upload_state_duration)
            else:
                # TODO convert this to a listener
                if (self.last_allreduce_block is not None) and (
                    (time.perf_counter() - self.all_reduce_start_time)
                    > (self.allreduce_timeout + self.upload_state_duration)
                ):
                    self.reload_state_event.set()
                elif (self.last_allreduce_block is None) and (
                    self.current_block - self.starting_block > 25
                ):
                    self.reload_state_event.set()
            time.sleep(10)

    def maybe_sync_and_reload(self):
        if not hasattr(self, "gloo_group"):
            return

        # This runs on the training/FSDP thread only.
        torch.cuda.set_device(self.local_rank)

        # Rank 0 publishes sync_flag; others send 0.
        sync_flag = (
            1 if (self.local_rank == 0 and self.reload_state_event.is_set()) else 0
        )
        sync = torch.tensor([sync_flag], device="cpu")
        dist.broadcast(sync, src=0, group=self.gloo_group)

        all_reduce_flag_tensor = torch.tensor([self.all_reduce_flag], device="cpu")
        dist.broadcast(all_reduce_flag_tensor, src=0, group=self.gloo_group)

        if (sync.item() == 0) and (all_reduce_flag_tensor == 0):
            return  # nothing to do
        else:
            self.reload_state_event.clear()

        # Reload on ALL ranks (simplest + avoids param rebroadcast complexities).
        # (If you truly want only rank 0 to load, you must then broadcast params,
        # which is more invasive.)
        self.logger.debug("Sync Reload Begin")

        if all_reduce_flag_tensor == 1:
            self.loop.run_until_complete(
                self.all_reduce_local(
                    distributed_training.protocol.AllReduce(
                        min_group_size=self.config.neuron.min_group_size,
                        timeout=self.allreduce_timeout,
                    )
                )
            )
        elif not self.all_reduce_success_status:
            block_list = self.model.config.block_list
            if self.local_progress.epoch > self.global_progress.epoch:
                self.logger.info(
                    f"Local Epoch {self.local_progress.epoch} Ahead Of Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
                )
                load_state_from_peer(
                    self,
                    epoch=self.global_progress.epoch,
                )
            else:
                load_state_from_peer(
                    self,
                    uid=self.uid,
                    epoch=self.global_progress.epoch,
                )
            self.model.config.block_list = block_list
            self.resume_training()
            self.all_reduce_success_status = True
        else:
            if self.last_allreduce_block is not None:
                self.load_state(reset_last_allreduce_block=True)
            else:
                self.starting_block = self.current_block
                self.load_state(reset_last_allreduce_block=False)

    def _process_training_batch(self, dataset):
        """Process a single training batch"""
        for i, batch in enumerate(dataset):
            inputs, _ = batch
            # TODO Can this be re-inrtoduced without hanging Rank1
            # if not self.training_active.is_set():
            #     break
            self.maybe_sync_and_reload()

            # Move to device
            inputs = inputs.to(self.local_rank)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(input_ids=inputs, labels=inputs)
                loss = outputs.loss / self.number_of_local_steps

            loss.backward()
            self.running_loss += loss.item() * self.number_of_local_steps
            self.batch_count += 1
            self.local_progress.loss = self.running_loss / self.batch_count
            self.local_progress.samples_accumulated += self.local_batch_size_train

            if (
                self.local_progress.samples_accumulated
                >= self.local_batch_size_train_effective
            ):
                self.logger.info(
                    f":training:  Outer Step: {self.local_progress.epoch} | "
                    f"Inner Step: {self.local_progress.inner_step} | "
                    f"Learning Rate: {self.inner_optimizer.param_groups[0]['lr']:.8f} | "
                    f"Average Loss: {self.local_progress.loss:.2f}"
                )

                self.event.update(
                    {
                        "train/outer_step": self.local_progress.epoch,
                        "train/inner_step": self.local_progress.inner_step,
                        "train/loss": self.local_progress.loss,
                        "train/learning_rate": self.inner_optimizer.param_groups[0][
                            "lr"
                        ],
                        "train/total_step": self.scheduler._step_count,
                    }
                )

                # Run inner optimizer step
                self.inner_optimizer_step()

                if (
                    self.local_progress.inner_step % self.config.neuron.upload_steps
                    == 0
                ):
                    # Upload model every x steps
                    self.start_background_upload(epoch=self.global_progress.epoch)
                    self.logger.info("start_background_upload compeleted")

    def inner_optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.inner_optimizer.step()

        self.inner_optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()

        self.local_progress.inner_step += 1

        self.running_loss = 0.0
        self.batch_count = 0

        self.local_progress.samples_accumulated = 0

    def upload_model(
        self,
        epoch,
        full_state=None,
        optim_sd=None,
        gradient_state=None,
        scheduler_state_dict=None,
        inner_optimizer_lr=None,
        archive=False,
    ):
        # self.pause_training()
        self.logger.info("Upload Model Start")
        # """Unified function to save and upload both model and optimizer state"""
        attempt = 0
        while attempt < self.model_upload_retry_limit:
            try:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                if self.master:
                    self.logger.info(
                        f"Saving model state dict with {len(full_state)} keys"
                    )
                    save_file(
                        full_state,
                        os.path.join(self.output_dir, "model.safetensors"),
                        metadata={"format": "pt"},
                    )
                    self.model.config.save_pretrained(self.output_dir)
                    self.logger.info(f"Model Saved")
                    del full_state
                    gc.collect()

                    # Save pseudo gradient state
                    torch.save(
                        gradient_state,
                        os.path.join(
                            self.output_dir,
                            "gradients.pt",
                        ),
                    )
                    del gradient_state

                # Save optimizer state
                optimizer_state = {
                    "optimizer_state_dict": optim_sd,
                    "learning_rate": inner_optimizer_lr,
                    "scheduler_state": scheduler_state_dict,
                    "epoch": epoch,
                }
                torch.save(
                    optimizer_state,
                    os.path.join(
                        self.output_dir,
                        f"inner_optimizer.rank{self.local_rank+1:04d}-of-{self.world_size}.pt",
                    ),
                )
                self.logger.info(f"Optimizer Saved")
                del optimizer_state

                if self.master:
                    # Reset model blocklist & keep local copy in case upload fails
                    block_list = self.model.config.block_list
                    self.model.config.block_list = []

                    self.logger.info(
                        f":upload: Uploading model and optimizer states to r2 bucket: {self.config.r2.bucket_name}"
                    )
                    self.upload_process = subprocess.Popen(
                        [
                            "python",
                            os.path.abspath(__file__).replace(
                                "neurons/miner.py",
                                "distributed_training/utils/upload_worker.py",
                            ),
                            self.config.r2.bucket_name,
                            self.config.r2.account_id,
                            self.config.r2.write.access_key_id,
                            self.config.r2.write.secret_access_key,
                            f"{__run__}.{self.local_progress.epoch}.{self.local_progress.inner_step}",
                            str(archive),
                        ]
                    )
                    while self.upload_process.poll() is None:
                        if not self.training_active.is_set():
                            self.upload_process.kill()
                            self.logger.info(
                                "Cancelling Ongoing Model Upload For AllReduce Operation"
                            )
                            self.model.config.block_list = (
                                block_list + self.model.config.block_list
                            )
                            return False
                        else:
                            time.sleep(5)

                    log_peerid_to_r2(self, prefix=f"epoch-{self.local_progress.epoch}/")
                    log_peerid_to_r2(self)

                    self.logger.info(
                        f"Successfully pushed new model state with tag {__run__}.{epoch}.{self.model.config.inner_step} to bucket: {self.config.r2.bucket_name}"
                    )

                return True

            except Exception as e:
                attempt += 1
                self.logger.warning(
                    f":error: Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
                )
                if attempt < self.model_upload_retry_limit:
                    time.sleep(self.model_upload_retry_delay)
                else:
                    self.logger.error(
                        "Maximum retry limit reached. Unable to upload state to HF Hub."
                    )
                    if self.master:
                        self.model.config.block_list = (
                            block_list + self.model.config.block_list
                        )
                    raise

        return False

    def start_background_upload(self, epoch, archive=False):
        """Starts a background upload of the model state, managing ongoing uploads."""
        # Check if upload_future is already being executed
        uploading = (
            1
            if (
                self.local_rank == 0
                and self.current_upload_future
                and not self.current_upload_future.done()
            )
            else 0
        )
        is_uploading = torch.tensor([uploading], device="cpu")
        dist.barrier(group=self.gloo_group)
        dist.broadcast(is_uploading, src=0, group=self.gloo_group)
        dist.barrier(group=self.gloo_group)

        if is_uploading.item() == 1:
            self.logger.info("Previous upload still in progress, skipping new upload")
            return  # nothing to do

        self.logger.info(f":memory: Saving model state locally for epoch {epoch}")
        self.model.config.inner_step = self.local_progress.inner_step

        opts = StateDictOptions(
            full_state_dict=True,  # gather a full (HF-style) state dict
            cpu_offload=True,  # offload to host RAM (no GPU OOM)
        )
        full_state = get_model_state_dict(self.model, options=opts)
        self.logger.info("Full State")

        o_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
        optim_sd = get_optimizer_state_dict(
            self.model, self.inner_optimizer, options=o_opts
        )
        self.logger.info(f"Extracted Optimizer & Model State Dict")

        gradient_state, _, _ = self.prepare_gradient_dict(quantize=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()

        # if self.master:
        if True:
            # Start new upload
            self.current_upload_future = self.upload_executor.submit(
                self.upload_model,
                epoch,
                full_state,
                optim_sd,
                gradient_state,
                self.scheduler.state_dict(),
                self.inner_optimizer.param_groups[0]["lr"],
                archive,
            )

            # Optional: Add callback to handle completion
            def upload_completed(future):
                try:
                    result = (
                        future.result()
                    )  # This will raise any exceptions that occurred
                    self.logger.info(
                        f"Model state upload completed with result: {result}"
                    )
                except Exception as e:
                    self.logger.error(f"Model state upload failed: {str(e)}")

            self.current_upload_future.add_done_callback(upload_completed)

        dist.barrier()
        del full_state, optim_sd, gradient_state
        gc.collect()

    def pause_training(self):
        """Pauses the continuous training loop"""
        self.training_active.clear()
        time.sleep(1)
        self.training_status = TrainingStatus.PAUSED
        self.logger.info(":warning:  Pausing continuous training.")

    def resume_training(self):
        """Resumes the continuous training loop"""
        self.training_active.set()
        self.training_status = TrainingStatus.RUNNING
        self.logger.info(":white_heavy_check_mark: Resuming continuous training.")

    async def fetch_training_data(self):
        """Async function to fetch training data"""
        attempt = 0
        while attempt < self.retry_limit:
            try:
                self.set_current_block_across_ranks()

                pages = await DatasetLoader.next_pages(
                    offset=self.current_block,
                    n_pages=5,
                    seed=self.uid + self.local_rank,
                )
                rng = np.random.default_rng(hash(self.uid) & 0xFFFFFFFF)
                rng.shuffle(pages)

                self.logger.debug(pages)

                dataset = await DatasetLoader.create(
                    batch_size=self.config.neuron.local_batch_size_train,
                    sequence_length=1024,
                    pages_info=pages,
                    tokenizer=self.tokenizer,
                )

                dataset_length = torch.tensor(len(dataset.buffer))
                dist.all_reduce(
                    dataset_length, op=dist.ReduceOp.MIN, group=self.gloo_group
                )
                dataset.buffer = dataset.buffer[:dataset_length]
                self.logger.debug("Dataset Buffer Length", len(dataset.buffer))

                return dataset
            except Exception as e:
                self.logger.error(f"Error fetching training data: {str(e)}")
                attempt += 1
                self.logger.warning(
                    f"Failed to fetch data, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay * attempt)  # Wait before the next retry
                else:
                    self.logger.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def _update_wandb_project(self):
        suffix = "_miners" if self.neuron_type == "MinerNeuron" else "_validators"
        self.config.neuron.wandb_project += suffix

    def _init_basic_components(self):
        """Initialize basic miner components and configurations."""
        setup_logging(self, config=self.config)

        # Core setup
        self.device = self.config.neuron.device
        init_dht(self)

        # Progress tracking
        self._init_progress_tracking()

        # Wandb setup
        if (not self.config.neuron.dont_wandb_log) and self.master:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Training components
        self._init_training_components()

        # Tracking metrics
        self._init_metrics_collection()

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

                # Create a background thread for periodic metric submission
                self.metrics_thread = threading.Thread(target=self._report_metrics_loop)
                self.metrics_thread.daemon = True
                self.metrics_thread.start()
                self.logger.info("Metrics tracking thread initialized successfully.")

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

    def _report_metrics_loop(self):
        """Periodically send metrics to InfluxDB"""
        while not self.stop_event.is_set():
            try:
                self._report_current_metrics()
            except Exception as e:
                self.logger.error(f"Error reporting metrics: {e}")
            time.sleep(30)  # Report every 30 seconds

    def _report_current_metrics(self):
        """Send current miner metrics to InfluxDB"""
        points = []

        # Training metrics
        point = (
            Point("training_metrics")
            .tag("miner_uid", str(self.uid))
            .tag("hotkey", self.wallet.hotkey.ss58_address)
            .tag("run_id", __run__)
            .tag("epoch", str(self.local_progress.epoch))
            .tag("inner_step", str(self.local_progress.inner_step))
            .field("loss", self.local_progress.loss)
            .field("samples_accumulated", self.local_progress.samples_accumulated)
            .field("samples_per_second", self.local_progress.samples_per_second)
        )
        points.append(point)

        # Resource metrics
        point = (
            Point("resource_metrics")
            .tag("miner_uid", str(self.uid))
            .tag("hotkey", self.wallet.hotkey.ss58_address)
            .tag("run_id", __run__)
            .field("cpu_percent", psutil.cpu_percent())
            .field("memory_percent", psutil.virtual_memory().percent)
            .field("gpu_utilization", self._get_gpu_utilization())
        )
        points.append(point)

        # Network metrics
        point = (
            Point("network_metrics")
            .tag("miner_uid", str(self.uid))
            .tag("hotkey", self.wallet.hotkey.ss58_address)
            .tag("run_id", __run__)
            .field("bandwidth", self._get_network_bandwidth())
        )
        points.append(point)

        # Metagraph metrics
        point = (
            Point("metagraph_metrics")
            .tag("miner_uid", str(self.uid))
            .tag("hotkey", self.wallet.hotkey.ss58_address)
            .tag("run_id", __run__)
            .field("stake", float(self.metagraph.stake[self.uid]))
            .field("trust", float(self.metagraph.trust[self.uid]))
            .field("consensus", float(self.metagraph.consensus[self.uid]))
            .field("incentive", float(self.metagraph.incentive[self.uid]))
            .field("emissions", float(self.metagraph.emission[self.uid]))
        )
        points.append(point)

        # Write points to InfluxDB
        self.influx_write_api.write(
            bucket=self.config.neuron.influxdb_bucket,
            org=self.config.neuron.influxdb_org,
            record=points,
        )

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            if self.device.startswith("cuda"):
                result = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                    .decode("utf-8")
                    .strip()
                )
                return float(result)
        except:
            pass
        return 0.0

    def _get_network_bandwidth(self):
        """Get network bandwidth usage in MB/s"""
        # Implement based on your system's network monitoring
        try:
            # This is a placeholder - implement actual bandwidth measurement
            return random.uniform(20, 30)  # MB/s
        except:
            return 0.0

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
        self.local_progress.inner_step = get_progress(self, "local")[1]

        if self.global_progress.epoch is None:
            self.logger.error(
                "Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )

    def _init_training_components(self):
        # Event tracking
        self.event = {}
        self.stop_event = threading.Event()

        # Training control
        self.training_active = threading.Event()
        self.training_active.set()

        # Queue and executor
        self.training_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="training_worker"
        )

        # Async components
        self.training_loop = asyncio.new_event_loop()
        self.training_lock = asyncio.Lock()

        # Status tracking
        self.training_status = TrainingStatus.STOPPED
        self.training_error = None

    def _init_model_components(self):
        """Initialize model-related components including tokenizer and optimizer settings."""
        self._init_tokenizer()
        self._setup_model_params()
        self._load_model()
        self._setup_training_params()
        self._load_gradient_compressors()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.neuron.global_tokenizer_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_model_params(self):
        # Optimizer settings
        self.learning_rate_maximum = 2.5e-4
        self.weight_decay = 0.1
        self.num_inner_steps = 500
        self.offload_optimizer = True

        # Upload settings
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 6

    def _load_model(self):
        # Load model and components
        load_state_from_peer(self, self.uid, self.local_progress.epoch)
        self.model.config.block_list = []
        cleanup_old_cache(self)

        # Setup upload executor
        self.upload_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model_upload"
        )
        self.current_upload_future = None
        self.upload_process = None
        # Sync and initialize handlers
        # TODO move to load_state_from_peers on first init only
        self._sync_with_global_model()

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
        self.last_allreduce_block = None

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

    def _init_network_components(self):
        """Initialize network and P2P components"""
        self.logger.info("Logging PeerID to chain")
        log_r2_to_chain(self)
        log_peerid_to_r2(self)

    def _sync_with_global_model(self):
        if self.master:
            if (
                self.config.neuron.global_model_name
                == self.config.neuron.local_model_name
            ):
                self.logger.warning(
                    "Your local miner_hf_repo_id set to the global model_name. This will harm your incentive. Set miner_hf_repo_id to a unique huggingface repo id."
                )

            # self.model.to("cpu")
            self.should_sync_model = (
                (self.local_progress.epoch is None)
                or (self.local_progress.epoch != self.global_progress.epoch)
                # or (
                #     compute_schema_hash(global_model.parameters())
                #     != compute_schema_hash(full_state.values())
                # )
            )
            # self.model.to(self.device)

        should_sync_model = (
            torch.tensor([1])
            if (self.master and self.should_sync_model)
            else torch.tensor([0])
        )
        dist.broadcast(should_sync_model, src=0, group=self.gloo_group)
        self.should_sync_model = True if should_sync_model[0].item() == 1 else False

        if self.should_sync_model:
            self.logger.info("Local model out of sync. Loading global model")
            load_state_from_peer(
                self, uid=self.master_uid, epoch=self.global_progress.epoch
            )

    @torch.no_grad()
    def compute_and_load_pseudo_grad_into_averager(self):
        """
        Rank 0 only:
        - Requests shards via RPC from other ranks one at a time.
        - Reconstructs parameter shard by shard on CPU.
        - Computes pseudo-gradients and loads into averager.
        """
        # assert self.local_rank == 0, "Only rank 0 should call this function!"

        opts = StateDictOptions(
            full_state_dict=True, cpu_offload=True
        )  # gather to CPU on rank 0
        full_state = get_model_state_dict(self.model, options=opts)
        if self.master:
            opt_parameters = [
                p for g in self.outer_optimizer.param_groups for p in g["params"]
            ]
            with self.grad_averager.get_tensors() as averaged_grads:
                for idx, (opt_param, averaged_grad, named_main_param) in enumerate(
                    zip(opt_parameters, averaged_grads, full_state.items())
                ):
                    _, submod = named_main_param
                    # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                    # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                    grad = opt_param.data - submod.detach().to(opt_param.device)
                    averaged_grad.copy_(grad, non_blocking=True)

    @torch.no_grad()
    def apply_optimizer_parameters(self):
        dist.barrier(group=self.gloo_group)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        opts_get = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        full_state = get_model_state_dict(self.model, options=opts_get)
        self.logger.info("Full state")

        if self.master:
            # Flatten optimizer params in the same order every time
            offloaded_parameters = [
                p for g in self.outer_optimizer.param_groups for p in g["params"]
            ]
            assert len(offloaded_parameters) == len(
                full_state
            ), f"mismatch: {len(offloaded_parameters)} vs {len(full_state)}"
            # full_state values are plain CPU tensors here (no DTensor)
            for (name, tensor), off_t in zip(full_state.items(), offloaded_parameters):
                assert isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                tensor.copy_(off_t, non_blocking=True)

        opts_set = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=True,
        )
        # Push back into the model (reshard). All ranks must enter.
        set_model_state_dict(
            model=self.model, model_state_dict=full_state, options=opts_set
        )

        del full_state
        gc.collect()
        torch.cuda.empty_cache()

    def create_pseudo_gradients(self):
        """compute pseudo gradient by subtracting the offloaded optimizer parameters with the main parameters and load them in the averager"""
        opt_parameters = [
            param
            for group in self.grad_averager.offloaded_optimizer.param_groups
            for param in group["params"]
        ]
        gradients = {}
        for opt_param, main_param, named_param in zip(
            opt_parameters,
            self.grad_averager.main_parameters,
            self.model.named_parameters(),
        ):
            # opt_param is the param that will be all_reduce, it is suppose to be on cpu
            # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
            gradients[named_param[0]] = opt_param.data - main_param.detach().to(
                opt_param.device
            )
        return gradients

    def prepare_gradient_dict(self, quantize: bool = False):
        """
        Prepares the gradient dictionary for sharing by compressing the
        momentum for each parameter and attaching metadata.

        Args:
            self (Miner): Instance of Miner containing model, scheduler, state_averager, gradient_averager, compressor, transformer and configs.
            quantize (bool): Whether to apply quantization during compression.

        Returns:
            tuple: (gradient, xshapes, totalks, transmitted) where:
                gradient (dict): Contains keys for each parameter's compressed gradients and metadata.
                xshapes (dict): The computed shapes for each parameter.
                totalks (dict): Total length information for each parameter.
        """
        gradient = {}
        xshapes = {}
        totalks = {}

        self.compute_and_load_pseudo_grad_into_averager()
        if self.master:
            with self.grad_averager.get_tensors() as averaged_grads:
                gradient_iterator = {
                    name: grad
                    for (name, parameter), grad in zip(
                        self.model.named_parameters(), averaged_grads
                    )
                }

            # Outer Optimizer LR
            lr = float(self.outer_optimizer.param_groups[0]["lr"])

            for n, p in gradient_iterator.items():
                # Apply momentum decay.
                self.error_feedback[n].mul_(self.config.neuron.momentum_decay)

                # Ensure the gradient is on the same device as the parameter.
                assert p is not None
                grad = p.detach().to(p.device)
                if self.error_feedback[n].device != p.device:
                    self.error_feedback[n] = (
                        self.error_feedback[n].detach().to(p.device)
                    )

                # Normal behavior for later iterations
                self.error_feedback[n].add_(grad, alpha=lr)

                # Compress momentum
                encoded = self.transformer.encode(
                    self.error_feedback[n], use_dct=self.config.neuron.use_dct
                )
                if quantize:
                    idxs, vals, xshape, totalk, quant_params = self.compressor.compress(
                        encoded, self.config.neuron.topk_compression, quantize
                    )
                else:
                    idxs, vals, xshape, totalk = self.compressor.compress(
                        encoded, self.config.neuron.topk_compression, quantize
                    )
                if totalk is None:
                    self.logger.info("totalk is None")
                del encoded  # Free the encoded tensor immediately

                if quantize:
                    # Estimate transmitted gradient
                    decompressed = self.compressor.decompress(
                        p, idxs, vals, xshape, totalk, quant_params
                    )
                else:
                    # Estimate transmitted gradient
                    decompressed = self.compressor.decompress(
                        p, idxs, vals, xshape, totalk, None
                    )
                transmit_grad = self.transformer.decode(
                    decompressed, use_dct=self.config.neuron.use_dct
                )
                del decompressed  # Free intermediate tensor

                self.error_feedback[n].sub_(transmit_grad)

                # Move compressed values to CPU to save GPU memory
                gradient[n + "idxs"] = (
                    idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
                )
                gradient[n + "vals"] = (
                    vals.cpu() if isinstance(vals, torch.Tensor) else vals
                )
                if quantize:
                    gradient[n + "quant_params"] = quant_params
                xshapes[n] = xshape
                totalks[n] = totalk

                del grad, transmit_grad, idxs, vals, xshape, totalk
                if quantize:
                    del quant_params
                gc.collect()

            # Delete graident iterator to free up memory
            del gradient_iterator
            torch.cuda.empty_cache()

            gradient["metadata"] = {
                "block": self.current_block,
                "inner_step": self.local_progress.inner_step,
                "outer_step": self.local_progress.epoch,
                "loss": self.local_progress.loss,
            }

        dist.barrier(device_ids=[self.local_rank])
        return gradient, xshapes, totalks

    def get_miner_info(self):
        return {
            "bittensor/block": self.metagraph.block.item(),
            "bittensor/stake": self.metagraph.stake[self.uid],
            "bittensor/trust": self.metagraph.trust[self.uid],
            "bittensor/consensus": self.metagraph.consensus[self.uid],
            "bittensor/incentive": self.metagraph.incentive[self.uid],
            "bittensor/emissions": self.metagraph.emission[self.uid],
        }

    async def is_alive(
        self, synapse: distributed_training.protocol.IsAlive
    ) -> distributed_training.protocol.IsAlive:
        self.logger.info("Responded to be Active")
        synapse.completion = "True"
        synapse.epoch = self.local_progress.epoch
        return synapse

    def start_continuous_training(self):
        """Starts continuous training using the ThreadPoolExecutor"""
        dist.barrier(device_ids=[self.local_rank])
        if self.training_status != TrainingStatus.RUNNING:
            self.training_status = TrainingStatus.RUNNING
            self.training_error = None
            self.training_executor.submit(self._training_worker)
            self.logger.info(
                ":white_heavy_check_mark: Starting continuous training worker"
            )

    def _training_worker(self):
        """Worker function that runs in the ThreadPoolExecutor"""

        asyncio.set_event_loop(self.training_loop)

        while not self.stop_event.is_set():
            try:
                # Wait if training is paused
                self.training_active.wait()

                self.logger.debug(":pages: Fetching fineweb-edu pages")
                dataset = self.training_loop.run_until_complete(
                    self.fetch_training_data()
                )

                # Wait if training is paused
                self.training_active.wait()

                if self.master:
                    self.model.config.block_list.append(self.current_block)

                self._process_training_batch(dataset)
            except Exception as e:
                self.logger.warning(f"Training Loop Failed with error: {e}")
                self.training_status = TrainingStatus.ERROR
                self.training_error = str(e)
                break

        self.training_status = TrainingStatus.STOPPED

    async def all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> distributed_training.protocol.AllReduce:
        self.logger.info("Received Main All Reduce Call")

        # Update gradient averager params to latest synapse values
        if synapse.timeout is not None:
            self.allreduce_timeout = synapse.timeout
        if synapse.min_group_size is not None:
            self.grad_averager.matchmaking_kwargs[
                "min_group_size"
            ] = synapse.min_group_size
        if synapse.request_timeout is not None:
            self.grad_averager.matchmaking_kwargs[
                "request_timeout"
            ] = synapse.request_timeout
        if synapse.allreduce_timeout is not None:
            self.grad_averager._allreduce_timeout = synapse.synapse.allreduce_timeout
        if synapse.next_chunk_timeout is not None:
            self.grad_averager.next_chunk_timeout = synapse.next_chunk_timeout
        if synapse.min_matchmaking_time is not None:
            self.grad_averager.matchmaking_kwargs[
                "min_matchmaking_time"
            ] = synapse.min_matchmaking_time

        self.all_reduce_flag = 1
        return synapse

    async def all_reduce_local(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> distributed_training.protocol.AllReduce:
        """Handle incoming all_reduce requests by pausing continuous training"""
        self.logger.info("Received Local All Reduce Call")
        self.all_reduce_start_time = time.perf_counter()
        initial_weights = None
        synapse.completion = True
        try:
            async with self.training_lock:
                # Cancel any ongoing upload
                if self.current_upload_future and not self.current_upload_future.done():
                    self.logger.info(
                        "Cancelling Ongoing Model Upload For AllReduce Operation"
                    )
                    self.current_upload_future.cancel()

                # Ensure training is paused
                self.pause_training()

                # Run inner optimizer step
                self.inner_optimizer_step()

                self.logger.info(":wait: Starting Compute Pseudo Gradients")
                self.compute_and_load_pseudo_grad_into_averager()
                self.logger.info(":wait: Finished Compute Pseudo Gradients")

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if self.master:
                    # Update gradient averager params to latest synapse values
                    if synapse.min_group_size is not None:
                        self.grad_averager.matchmaking_kwargs[
                            "min_group_size"
                        ] = synapse.min_group_size
                    if synapse.request_timeout is not None:
                        self.grad_averager.matchmaking_kwargs[
                            "request_timeout"
                        ] = synapse.request_timeout
                    if synapse.allreduce_timeout is not None:
                        self.grad_averager._allreduce_timeout = (
                            synapse.synapse.allreduce_timeout
                        )
                    if synapse.next_chunk_timeout is not None:
                        self.grad_averager.next_chunk_timeout = (
                            synapse.next_chunk_timeout
                        )
                    if synapse.min_matchmaking_time is not None:
                        self.grad_averager.matchmaking_kwargs[
                            "min_matchmaking_time"
                        ] = synapse.min_matchmaking_time

                    try:
                        self.logger.info("All Reduce Start")
                        # Run allreduce with proper timeout
                        (
                            synapse,
                            initial_weights,
                        ) = await self.avg_handler.run_miner_allreduce(
                            synapse,
                            self.local_progress,
                            self.all_reduce_start_time,
                            self.current_block,
                            # bandwidth
                        )
                        self.logger.info("All Reduce Finish")
                        if not synapse.completion:
                            raise Exception("AllReduce Failed, Loading Latest State")
                    except Exception as e:
                        self.logger.info(f"All Reduce Failed with error: {e}")
                        synapse.completion = False

        except Exception as e:
            synapse.completion = False
            raise Exception(f"Unexpected error during AllReduce: {str(e)}") from e

        finally:
            if not gloabl_dist_checkpoint(synapse.completion, self.gloo_group):
                self.all_reduce_flag = 0
                self.all_reduce_success_status = False
                self.resume_training()
                self.maybe_sync_and_reload()
                self.logger.info(f"All Reduce Failed At Checkpoint 1")
                return synapse

            # Normalize averaged gradients
            try:
                self.logger.info(
                    f"Initial Weights NORM: {torch.norm(torch.cat([p.data.view(-1) for p in self.model.parameters()]))}"
                )
                # Perform offloaded outer optimization steps
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            except Exception:
                synapse.completion = False

            if self.master:
                self.logger.info("Outer Optimizer Step Started")
                for i, group in enumerate(self.outer_optimizer.param_groups):
                    for p in group["params"]:
                        self.logger.info(
                            f"group {i} param {p.shape} grad mean={p.grad.float().mean().item()} p mean={p.float().mean().item()}"
                        )
                        break
                self.outer_optimizer.step()
                self.logger.info("Outer Optimizer Step Finisheds")
                for i, group in enumerate(self.outer_optimizer.param_groups):
                    for p in group["params"]:
                        self.logger.info(
                            f"group {i} param {p.shape} grad mean={p.grad.float().mean().item()} p mean={p.float().mean().item()}"
                        )
                        break

            if not gloabl_dist_checkpoint(synapse.completion, self.gloo_group):
                self.all_reduce_flag = 0
                self.all_reduce_success_status = False
                self.resume_training()
                self.maybe_sync_and_reload()
                self.logger.info(f"All Reduce Failed At Checkpoint 2")
                return synapse

            self.logger.info(f"Apply opt params")
            self.apply_optimizer_parameters()

            self.logger.info(":white_heavy_check_mark: Finished Outer Optimizer Step.")

            if self.master:
                # Validate weight updates
                await self.avg_handler._validate_weight_update(
                    initial_weights, self.current_block
                )

            self.logger.info(
                f"Initial Weights NORM: {torch.norm(torch.cat([p.data.view(-1) for p in self.model.parameters()]))}"
            )

            if not gloabl_dist_checkpoint(synapse.completion, self.gloo_group):
                self.all_reduce_flag = 0
                self.all_reduce_success_status = False
                self.resume_training()
                self.maybe_sync_and_reload()
                self.logger.info(f"All Reduce Failed At Checkpoint 3")
                return synapse

            # Reset inner_step and update epoch
            self.local_progress.samples_accumulated = 0
            self.local_progress.inner_step = 0
            self.local_progress.epoch += 1
            self.set_current_block_across_ranks()
            self.last_allreduce_block = self.current_block
            self.logger.info("AllReduce Operation Finished Succesfully")
            self.start_background_upload(
                epoch=self.local_progress.epoch,
                archive=True,
            )
            self.all_reduce_flag = 0
            self.reload_state_event.clear()

            # Resume training when done
            self.resume_training()
            self.maybe_sync_and_reload()
            return synapse

    async def blacklist_base(self, synapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.AllReduce): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        hotkey = synapse.dendrite.hotkey
        synapse_type = type(synapse).__name__

        uid = None
        axon = None
        for _uid, _axon in enumerate(self.metagraph.axons):
            if _axon.hotkey == hotkey:
                uid = _uid
                axon = _axon
                break

        if uid is None:
            self.logger.trace(
                f"Blacklisting unrecognized hotkey: {synapse.dendrite.hotkey}"
            )
            return (
                True,
                f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
            )

        if self.config.blacklist.force_validator_permit and (
            not self.config.blacklist.allow_non_registered
        ):
            # Check stake if uid is recognize
            tao = self.metagraph.neurons[uid].stake.tao
            if tao < self.config.neuron.vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a low stake {synapse_type} request: {tao} < {self.config.neuron.vpermit_tao_limit} from {hotkey}",
                )

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            self.logger.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        self.logger.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def blacklist_is_alive(
        self, synapse: distributed_training.protocol.IsAlive
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        self.logger.debug(blacklist[1])
        return blacklist

    async def blacklist_all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        self.logger.debug(blacklist[1])
        return blacklist


# This is the main function, which runs the miner.
if __name__ == "__main__":
    Miner().run()
