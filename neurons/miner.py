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
from distributed_training.data.r2_dataset import R2DatasetLoader 
from distributed_training.utils.chain import log_peerid_to_chain
from distributed_training.utils.misc import (
    init_dht,
    load_wandb,
)
from distributed_training.utils.logger import setup_logging
from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    get_global_epoch,
    get_local_inner_step,
)
from distributed_training.utils.state_loader import (
    cleanup_old_cache,
    load_model_optimizer_gradient_averager,
    load_state_from_peer,
)
from distributed_training.utils.compression import TransformDCT, CompressDCT


class Miner(BaseMinerNeuron):
    def __init__(self, config=None, use_r2=False):
        super(Miner, self).__init__(config=config)
        self._update_wandb_project()
        self._init_basic_components()
        self._init_model_components()
        self._init_network_components()
        self.data_loader = R2DatasetLoader if use_r2 else DatasetLoader

    def _update_wandb_project(self):
        suffix = "_miners" if self.neuron_type == "MinerNeuron" else "_validators"
        self.config.neuron.wandb_project += suffix

    def _init_basic_components(self):
        """Initialize basic miner components and configurations."""
        setup_logging(self, config=self.config)

        # Core setup
        self.device = self.config.neuron.device
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        init_dht(self)

        # Progress tracking
        self._init_progress_tracking()

        # Wandb setup
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Training components
        self._init_training_components()

        # Tracking metrics
        self._init_metrics_collection()

    def _init_metrics_collection(self):
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
            self.logger.info("InfluxDB client and write_api initialized successfully.")

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
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
            inner_step=0,
            loss=0.0,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        self.global_progress.epoch = get_global_epoch(self)
        self.local_progress.epoch = self.global_progress.epoch
        self.local_progress.inner_step = get_local_inner_step(self)

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

        # Save directory
        self.output_dir = self.config.neuron.local_model_name.split("/")[-1]

        # Create Tag Deletion Queue & Thread
        self.tag_deletion_queue = Queue()
        self.tag_deletion_thread = threading.Thread(target=self.delete_tags)
        self.tag_deletion_thread.start()

    def delete_tags(self):
        while True:
            if self.tag_deletion_queue.qsize() <= 0:
                time.sleep(60)
            else:
                tag_name = self.tag_deletion_queue.get()
                try:
                    # Update tag for this version
                    delete_tag(
                        self.config.neuron.local_model_name,
                        repo_type="model",
                        tag=tag_name,
                    )
                    self.logger.info(f"Succesfully deleted tag {tag_name}")
                except Exception as e:
                    self.logger.info(f"Failed to delete tag {tag_name} with error {e}")
                time.sleep(30)

    def _init_model_components(self):
        """Initialize model-related components including tokenizer and optimizer settings."""
        self._init_tokenizer()
        self._setup_model_params()
        self._load_model()
        self._setup_training_params()
        self._load_gradient_compressors()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.neuron.global_model_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_model_params(self):
        # Optimizer settings
        self.learning_rate_maximum = 4e-4
        self.weight_decay = 0.1
        self.num_inner_steps = 500
        self.offload_optimizer = True

        # Upload settings
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 6

    def _load_model(self):
        # Load model and components
        load_model_optimizer_gradient_averager(
            self, self.config.neuron.local_model_name, self.local_progress.epoch
        )
        self.model.config.block_list = []
        cleanup_old_cache(self, repo_id=self.config.neuron.local_model_name)

        # Setup upload executor
        self.upload_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model_upload"
        )
        self.current_upload_future = None
        self.upload_process = None

        # Sync and initialize handlers
        self._sync_with_global_model()

    def _setup_training_params(self):
        self.local_batch_size_train = self.config.neuron.local_batch_size_train
        self.local_batch_size_train_effective = (
            self.config.neuron.local_batch_size_train_effective
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
        # Init compression
        self.transformer = TransformDCT(
            self.model, target_chunk=self.config.neuron.target_chunk
        )
        self.compressor = CompressDCT(
            use_quantization=True,
            quantization_bins=self.config.neuron.quantization_bins,
            quantization_range=self.config.neuron.quantization_range,
        )
        self.xshapes = {}
        self.totalks = {}
        self.error_feedback = {}
        self.owned_params = set()
        for n, p in self.model.named_parameters():
            self.owned_params.add(n)
            self.error_feedback[n] = torch.zeros_like(p, device=self.device)
            _, _, xshape, totalk = self.compressor.compress(
                self.transformer.encode(
                    torch.zeros_like(p), use_dct=self.config.neuron.use_dct
                ),
                self.config.neuron.topk_compression,
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

    def _init_network_components(self):
        """Initialize network and P2P components"""
        self.logger.info("Logging PeerID to chain")
        log_peerid_to_chain(self)

    def _sync_with_global_model(self):
        global_model = AutoModelForCausalLM.from_pretrained(
            self.config.neuron.global_model_name,
            revision=f"{__run__}.{self.global_progress.epoch}.0",
            trust_remote_code=False,
        )

        if self.config.neuron.global_model_name == self.config.neuron.local_model_name:
            self.logger.warning(
                "Your local miner_hf_repo_id set to the global model_name. This will harm your incentive. Set miner_hf_repo_id to a unique huggingface repo id."
            )

        self.model.to("cpu")
        should_sync_model = (
            (self.local_progress.epoch is None)
            or (self.local_progress.epoch != self.global_progress.epoch)
            or (
                compute_schema_hash(global_model.parameters())
                != compute_schema_hash(self.model.parameters())
            )
        )
        self.model.to(self.device)

        if should_sync_model:
            del global_model
            gc.collect()
            torch.cuda.empty_cache()
            load_state_from_peer(self, epoch=self.global_progress.epoch)
            self.start_background_upload(
                epoch=self.global_progress.epoch,
            )
        else:
            del global_model
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

        # Outer Optimizer LR
        lr = float(self.state_averager.optimizer.param_groups[0]["lr"])

        gradient_iterator = self.create_pseudo_gradients()
        for n, p in gradient_iterator.items():
            # Apply momentum decay.
            self.error_feedback[n].mul_(self.config.neuron.momentum_decay)

            # Ensure the gradient is on the same device as the parameter.
            assert p is not None
            grad = p.detach().to(p.device)
            if self.error_feedback[n].device != p.device:
                self.error_feedback[n] = self.error_feedback[n].detach().to(p.device)

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
                bt.logging.info("totalk is None")
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

        return gradient, xshapes, totalks

    def upload_model(self, epoch):
        """Unified function to save and upload both model and optimizer state"""
        if not repo_exists(self.config.neuron.local_model_name, repo_type="model"):
            try:
                create_repo(
                    self.config.neuron.local_model_name,
                    repo_type="model",
                    private=False,
                )
                self.logger.info(
                    f"Created new repository: {self.config.neuron.local_model_name}"
                )
            except Exception as e:
                self.logger.error(f"Failed to create repository: {str(e)}")
                raise

        attempt = 0
        while attempt < self.model_upload_retry_limit:
            # Check if training is paused (i.e. all_reduce is happening)
            if not self.training_active.is_set():
                self.logger.info("Upload Cancelled Due To AllReduce Operation")
                return False
            try:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                self.logger.info(
                    f":memory: Saving model state locally for epoch {epoch}"
                )
                self.model.config.inner_step = self.local_progress.inner_step
                self.model.save_pretrained(os.path.join(self.output_dir))

                # Reset model blocklist & keep local copy in case upload fails
                block_list = self.model.config.block_list
                self.model.config.block_list = []

                # Save optimizer state
                optimizer_state = {
                    "optimizer_state_dict": self.inner_optimizer.state_dict(),
                    "learning_rate": self.inner_optimizer.param_groups[0]["lr"],
                    "scheduler_state": self.scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    optimizer_state,
                    os.path.join(
                        self.output_dir,
                        "inner_optimizer.pt",
                    ),
                )

                # Create and save pseudo gradient state
                gradient_state, _, _ = self.prepare_gradient_dict(quantize=False)
                torch.save(
                    gradient_state,
                    os.path.join(
                        self.output_dir,
                        "gradients.pt",
                    ),
                )

                self.logger.info(
                    f":upload: Uploading model and optimizer states to repo: {self.config.neuron.local_model_name}"
                )
                commit_message = f"Run {__run__}. Outer Step {epoch}. Inner Step {self.local_progress.inner_step}."
                self.upload_process = subprocess.Popen(
                    [
                        "python",
                        os.path.abspath(__file__).replace(
                            "neurons/miner.py",
                            "distributed_training/utils/upload_worker.py",
                        ),
                        self.config.neuron.local_model_name,
                        self.output_dir,
                        commit_message,
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

                refs = list_repo_refs(
                    self.config.neuron.local_model_name, repo_type="model"
                )
                for tag in refs.tags:
                    if (tag.name == "None") or (
                        tag.name == f"{__run__}.{epoch}.{self.model.config.inner_step}"
                    ):
                        # Update tag for this version
                        delete_tag(
                            self.config.neuron.local_model_name,
                            repo_type="model",
                            tag=tag.name,
                        )
                        time.sleep(30)
                    elif (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                        and (int(tag.name.split(".")[1]) > epoch)
                    ):
                        self.tag_deletion_queue.put(tag.name)
                # Create new tag for this version
                create_tag(
                    self.config.neuron.local_model_name,
                    repo_type="model",
                    tag=f"{__run__}.{epoch}.{self.model.config.inner_step}",
                    tag_message=commit_message,
                )
                # Cleanup old cache
                cleanup_old_cache(
                    self,
                    repo_id=self.config.neuron.local_model_name,
                    current_revision=None,
                )

                self.logger.info(
                    f"Successfully pushed new model state with tag {__run__}.{epoch}.{self.model.config.inner_step} to repo: {self.config.neuron.local_model_name}"
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
                    self.model.config.block_list = (
                        block_list + self.model.config.block_list
                    )
                    raise

        return False

    def start_background_upload(self, epoch):
        """Starts a background upload of the model state, managing ongoing uploads."""
        # If there's an ongoing upload, check if it's done
        if self.current_upload_future and not self.current_upload_future.done():
            self.logger.info("Previous upload still in progress, skipping new upload")
            return

        # Start new upload
        self.current_upload_future = self.upload_executor.submit(
            self.upload_model, epoch
        )

        # Optional: Add callback to handle completion
        def upload_completed(future):
            try:
                result = future.result()  # This will raise any exceptions that occurred
                self.logger.info(f"Model state upload completed with result: {result}")
            except Exception as e:
                self.logger.error(f"Model state upload failed: {str(e)}")

        self.current_upload_future.add_done_callback(upload_completed)

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
        if self.training_status != TrainingStatus.RUNNING:
            self.training_status = TrainingStatus.RUNNING
            self.training_error = None
            self.training_executor.submit(self._training_worker)
            self.logger.info(
                ":white_heavy_check_mark: Starting continuous training worker"
            )

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
                pages = await self.data_loader.next_pages(
                    offset=self.current_block,
                    n_pages=35,
                    seed=self.uid,
                )
                random.seed(self.uid)
                random.shuffle(pages)

                dataset = await self.data_loader.create(
                    batch_size=self.config.neuron.local_batch_size_train,
                    sequence_length=1024,
                    pages_info=pages,
                    tokenizer=self.tokenizer,
                )

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

    def _training_worker(self):
        """Worker function that runs in the ThreadPoolExecutor"""

        asyncio.set_event_loop(self.training_loop)

        while not self.stop_event.is_set():
            try:
                # Wait if training is paused
                self.training_active.wait()

                # Periodic model upload
                if (
                    len(self.model.config.block_list)
                    >= self.config.neuron.target_n_blocks
                ):
                    self.start_background_upload(
                        epoch=self.local_progress.epoch,
                    )

                self.logger.debug(":pages: Fetching fineweb-edu pages")
                dataset = self.training_loop.run_until_complete(
                    self.fetch_training_data()
                )

                # Wait if training is paused
                self.training_active.wait()

                self.model.config.block_list.append(self.current_block)
                self._process_training_batch(dataset)
            except Exception as e:
                self.logger.warning(f"Training Loop Failed with error: {e}")
                self.training_status = TrainingStatus.ERROR
                self.training_error = str(e)
                break

        self.training_status = TrainingStatus.STOPPED

    def _process_training_batch(self, dataset):
        """Process a single training batch"""

        for inputs, labels in dataset:
            if not self.training_active.is_set():
                break

            # Move to device
            inputs = inputs.to(self.device)

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

    def inner_optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.inner_optimizer.step()

        self.scheduler.step()

        self.inner_optimizer.zero_grad()

        self.local_progress.inner_step += 1

        self.running_loss = 0.0
        self.batch_count = 0

        self.local_progress.samples_accumulated = 0

    async def all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> distributed_training.protocol.AllReduce:
        """Handle incoming all_reduce requests by pausing continuous training"""
        self.logger.info("Received All Reduce Call")
        self.all_reduce_start_time = time.perf_counter()
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
                    self.grad_averager.next_chunk_timeout = synapse.next_chunk_timeout
                if synapse.min_matchmaking_time is not None:
                    self.grad_averager.matchmaking_kwargs[
                        "min_matchmaking_time"
                    ] = synapse.min_matchmaking_time

                try:
                    # Run allreduce with proper timeout
                    synapse = await self.avg_handler.run_miner_allreduce(
                        synapse,
                        self.local_progress,
                        self.all_reduce_start_time,
                        self.current_block,
                        # bandwidth
                    )
                    if not synapse.completion:
                        raise Exception("AllReduce Failed, Loading Latest State")
                except Exception as e:
                    self.logger.info(f"All Reduce Failed with error: {e}")
                    synapse.completion = False

        except Exception as e:
            synapse.completion = False
            raise Exception(f"Unexpected error during AllReduce: {str(e)}") from e

        finally:
            # Update epoch if all_reduce was succsefull
            if synapse.completion is True:
                # Reset inner_step and update epoch
                self.local_progress.samples_accumulated = 0
                self.local_progress.inner_step = 0
                self.local_progress.epoch += 1
                self.last_allreduce_block = self.current_block
                self.logger.info("AllReduce Operation Finished Succesfully")
                self.start_background_upload(
                    epoch=self.local_progress.epoch,
                )
                # Resume training when done
                self.resume_training()
            else:
                self.all_reduce_success_status = False

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
