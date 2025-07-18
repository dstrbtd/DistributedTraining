import copy
import gc
import os
import requests
import subprocess
import pytz
import sys
import shutil
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Optional

import bittensor as bt
import hivemind
import psutil
import torch
from memory_profiler import profile
from datetime import datetime

from hivemind.compression import deserialize_torch_tensor
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from huggingface_hub import (
    create_tag,
    hf_hub_download,
    list_repo_refs,
    list_repo_files,
    scan_cache_dir,
    upload_folder,
)
from huggingface_hub.utils import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    EntryNotFoundError,
)
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)

from distributed_training import __run__
from distributed_training.averaging.averagers import DTGradAverager, DTStateAverager
from distributed_training.utils.progress_tracker import (
    get_global_epoch,
    get_local_inner_step,
    get_min_local_inner_Step,
)
from distributed_training.averaging.avg_handler import AveragingHandler
from huggingface_hub import list_repo_commits

hivemind_logger = get_logger(__name__)


class ModelLoadingManager:
    def __init__(self):
        self.loading_lock = threading.Lock()
        self._is_loading = False
        self._last_loaded_epoch = None

    @property
    def is_loading(self):
        with self.loading_lock:
            return self._is_loading

    @property
    def last_loaded_epoch(self):
        with self.loading_lock:
            return self._last_loaded_epoch

    def set_loading_state(self, is_loading, epoch=None):
        with self.loading_lock:
            self._is_loading = is_loading
            if not is_loading and epoch is not None:
                self._last_loaded_epoch = epoch


class FastModelLoader:
    def __init__(self, model_name: str, cache_dir: str = None):
        """
        Initialize the fast model loader with HF downloader integration.

        Args:
            model_name (str): The HuggingFace model name (e.g., 'organization/model-name')
            cache_dir (str, optional): Directory to store downloaded files. Defaults to HF cache.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self._downloaded_files = {}  # Cache of downloaded files

    def download_files(self, revision: str = None, files: list = None):
        """
        Download files using hfdownloader.

        Args:
            revision (str, optional): Git revision/epoch number
            files (list, optional): List of specific files to download with patterns

        Returns:
            str: Path to downloaded files
        """
        # Generate cache key
        cache_key = f"{revision}_{','.join(files) if files else 'default'}"

        # Check if we already downloaded these files
        if cache_key in self._downloaded_files:
            return self._downloaded_files[cache_key]

        model_path = os.path.join(self.cache_dir, self.model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        cmd = [
            "hfdownloader",
            "-r",
            self.model_name,
            "download",
            "-c",
            "10",
            "-y",
        ]

        if revision:
            cmd.extend(["-b", revision])

        # Add file patterns if specified, otherwise default to both model and optimizer
        if files:
            for file_pattern in files:
                cmd.extend(["-f", f"{file_pattern}"])
        else:
            cmd.extend(
                [
                    "-f",
                    "*.safetensors",
                    "-f",
                    "optimizer.pt",
                    "--skip-verify",
                ]
            )

        bt.logging.debug(f"Executing hfdownloader command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=model_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                },  # Force Python unbuffered output
            )

            # Use select to handle both stdout and stderr
            import select

            outputs = [process.stdout, process.stderr]
            while True:
                # Wait for output on either stdout or stderr
                readable, _, _ = select.select(outputs, [], [])

                for output in readable:
                    line = output.readline()
                    if line:
                        # Don't buffer the print
                        print(line.rstrip(), flush=True)

                # Check if process has finished
                if process.poll() is not None:
                    break

            # Get any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                print(remaining_stdout.rstrip(), flush=True)
            if remaining_stderr:
                print(
                    f"Error: {remaining_stderr.rstrip()}", file=sys.stderr, flush=True
                )

            if process.returncode != 0:
                raise RuntimeError(
                    f"hfdownloader failed with return code {process.returncode}"
                )

        except Exception as e:
            bt.logging.error(f"Download failed! Error: {str(e)}")
            raise RuntimeError(f"hfdownloader failed: {str(e)}")

        return model_path

    def load_model_and_optimizer(self, epoch: int = None):
        """
        Load both model and optimizer states in a single download operation.

        Args:
            epoch (int, optional): Epoch number for specific revision

        Returns:
            tuple: (model_state_dict, optimizer_state_dict)
        """
        revision = str(epoch) if epoch is not None else None

        # Download both model and optimizer files in one go
        model_path = self.download_files(revision=revision)

        # Load model state
        model_files = list(Path(model_path).rglob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")

        bt.logging.info(f"Loading model state from: {[f.name for f in model_files]}")

        state_dict = {}
        for model_file in model_files:
            from safetensors.torch import load_file

            state = load_file(model_file)
            state_dict.update(state)

        # Load optimizer state
        optimizer_file = Path(model_path) / "optimizer.pt"
        if not optimizer_file.exists():
            raise FileNotFoundError(f"Optimizer state not found at {optimizer_file}")

        bt.logging.info(f"Loading optimizer state from: {optimizer_file}")
        optimizer_state = torch.load(str(optimizer_file), map_location="cpu")

        return state_dict, optimizer_state


def check_model_exists(repo_id: str, revision: Optional[str] = None) -> bool:
    try:
        if revision and revision != "None":
            list_repo_files(repo_id, revision=revision)
        else:
            list_repo_files(repo_id)
        return True
    except Exception as e:
        bt.logging.info(f"Model or revision check failed with error: {e}")
        return False


# @profile
def load_model_optimizer_gradient_averager(
    self,
    local_model_name,
    epoch,
    reload_inner_optimizer=True,
    reload_outer_optimizer=True,
    revision=None,
    use_fallback_model=True,
    reset_block_list=True,
):
    """
    Pytorch currently have an ongoing issue with memory leaks:
    https://github.com/pytorch/pytorch/issues/64043. To mitigate
    against this for now gc.collect() is run after each component
    with optimizers and state averagers are deleted.
    """
    bt.logging.debug(
        f"CPU Memory Before Loading State {psutil.virtual_memory().available / 10**9} GB"
    )
    global_model_name = self.config.neuron.global_model_name
    self.global_model_config = AutoConfig.from_pretrained(
        global_model_name, trust_remote_code=False
    )
    if use_fallback_model:
        model_name_list = [local_model_name, global_model_name]
    else:
        model_name_list = [local_model_name]

    if (revision is None) and (local_model_name != global_model_name):
        revision = f"{__run__}.{epoch}.{self.local_progress.inner_step}"
    elif (revision is None) and (local_model_name == global_model_name):
        revision = f"{__run__}.{epoch}.0"

    # Delete Gradient and State Averagers
    if hasattr(self, "state_averager"):
        self.grad_averager.shutdown()
        while self.grad_averager.is_alive():
            time.sleep(1)

        del self.grad_averager.main_parameters
        del self.grad_averager.offloaded_optimizer
        del self.grad_averager._averaged_tensors
        del self.grad_averager
        gc.collect()
        torch.cuda.empty_cache()

        self.state_averager.shutdown()
        while self.state_averager.is_alive():
            time.sleep(1)

        del self.state_averager.optimizer.param_groups
        del self.state_averager.optimizer
        del self.state_averager.main_parameters
        del self.state_averager._averaged_tensors
        del self.state_averager

        gc.collect()
        torch.cuda.empty_cache()
        bt.logging.info("Deleted State Averager and Gradient Averager")

    # Delete existing averag handler
    if hasattr(self, "avg_handler"):
        del self.avg_handler.model
        del self.avg_handler.inner_optimizer
        del self.avg_handler.grad_averager
        del self.avg_handler.state_averager
        del self.avg_handler
        gc.collect()
        torch.cuda.empty_cache()
        bt.logging.info("Deleted Average Handler")

    for model_name in model_name_list:
        optimizer_state = None
        # Load Model & Inner Optimizer
        try:
            if model_name == global_model_name:
                revision = ".".join(revision.split(".")[:-1] + ["0"])
            if not check_model_exists(
                model_name,
                revision=revision,
            ):
                continue

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=False,
            )
            bt.logging.info(
                f"Successfully Loaded Model From {model_name} With Revision {revision}"
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.config.block_list = []
            self.local_progress.inner_step = (
                self.model.config.inner_step
                if "inner_step" in self.model.config.__dict__
                else 0
            )
            if (model_name == global_model_name) and (
                epoch == self.global_progress.epoch
            ):
                self.allreduce_status_dict = (
                    self.model.config.all_reduce_scores
                    if "all_reduce_scores" in self.model.config.__dict__
                    else {}
                )

            if reload_inner_optimizer:
                # Delete existing inner optimizer
                if hasattr(self, "inner_optimizer"):
                    for i in self.inner_optimizer.param_groups[0]["params"]:
                        del i
                        gc.collect()
                        torch.cuda.empty_cache()
                    del self.inner_optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    bt.logging.info("Deleted Inner Optimizer")

                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate_maximum,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                )
                bt.logging.info(f"Loaded Inner Optimizer")

                self.scheduler = get_cosine_schedule_with_warmup(
                    self.inner_optimizer,
                    num_warmup_steps=1000,
                    num_training_steps=88000,
                )

                optimizer_state = torch.load(
                    hf_hub_download(
                        repo_id=model_name,
                        filename="inner_optimizer.pt",
                        revision=revision,
                    ),
                    weights_only=True,
                    map_location="cpu",
                )

                # Load optimizer state if available
                if "optimizer_state_dict" in optimizer_state:
                    self.inner_optimizer.load_state_dict(
                        optimizer_state["optimizer_state_dict"]
                    )
                if "learning_rate" in optimizer_state:
                    for group in self.inner_optimizer.param_groups:
                        group["lr"] = optimizer_state["learning_rate"]
                if "scheduler_state" in optimizer_state:
                    self.scheduler.load_state_dict(optimizer_state["scheduler_state"])
                bt.logging.info(
                    f"Successfully Loaded Inner Optimizer State From {model_name} For Revision {revision}"
                )

                break

        except Exception as e:
            if model_name == model_name_list[-1]:
                raise Exception(f"Failed to load model despite repo existing: {str(e)}")
            else:
                bt.logging.info(f"Failed to load model despite repo existing: {str(e)}")

        finally:
            if isinstance(optimizer_state, dict):
                keys = list(optimizer_state.keys())
                for k in keys:
                    del optimizer_state[k]
                    gc.collect()
            del optimizer_state
            gc.collect()
            torch.cuda.empty_cache()

    # Set outer optimizer
    self.outer_optimizer = partial(torch.optim.SGD, lr=0.7, momentum=0.9, nesterov=True)

    # Load a new state averager
    self.state_averager = DTStateAverager(
        dht=self.dht,
        prefix=f"{self.config.neuron.run_id}_state_averager",
        optimizer=self.outer_optimizer,
        params=self.model.parameters(),
        initialize_optimizer=True,
        offload_optimizer=self.offload_optimizer,
        custom_gradients=self.offload_optimizer,
        min_group_size=self.config.neuron.min_group_size,
        min_matchmaking_time=30.0,
        request_timeout=10.0,
        next_chunk_timeout=45.0,
        allreduce_timeout=self.allreduce_timeout - 30.0 - 15.0,
        start=True,
    )
    bt.logging.info("Successfully Loaded Gradient Averager")

    # Load a new gradient averager
    self.grad_averager = DTGradAverager(
        dht=self.dht,
        main_parameters=self.state_averager.main_parameters,
        offloaded_optimizer=self.state_averager.optimizer,
        prefix=f"{self.config.neuron.run_id}_grad_averager",
        compression=hivemind.Uniform8BitQuantization(),
        state_compression=hivemind.Uniform8BitQuantization(),
        min_group_size=self.config.neuron.min_group_size,
        min_matchmaking_time=30.0,
        request_timeout=10.0,
        next_chunk_timeout=45.0,
        allreduce_timeout=self.allreduce_timeout - 30.0 - 15.0,
        start=True,
    )
    bt.logging.info("Successfully Loaded State Averager")

    if reload_outer_optimizer:
        optimizer_state = None
        try:
            optimizer_state = torch.load(
                hf_hub_download(
                    repo_id=global_model_name,
                    filename="outer_optimizer.pt",
                    revision=".".join(revision.split(".")[:-1] + ["0"]),
                ),
                weights_only=True,
                map_location="cpu",
            )

            # Load optimizer state if available
            if "optimizer_state_dict" in optimizer_state:
                self.state_averager.optimizer.load_state_dict(
                    optimizer_state["optimizer_state_dict"]
                )

            bt.logging.info(
                f"Successfully Loaded Outer Optimizer State From {global_model_name} For Revision {'.'.join(revision.split('.')[:-1] + ['0'])}"
            )

        except Exception as e:
            bt.logging.warning(
                f"No optimizer state found or failed to load: {str(e)}. Initializing fresh optimizer."
            )

        finally:
            if isinstance(optimizer_state, dict):
                keys = list(optimizer_state.keys())
                for k in keys:
                    del optimizer_state[k]
                    gc.collect()
            del optimizer_state
            gc.collect()
            torch.cuda.empty_cache()

    self.avg_handler = AveragingHandler(
        self.model,
        self.inner_optimizer,
        self.grad_averager,
        self.state_averager,
        self.retry_limit,
        self.retry_delay,
        self.uid,
        self.config.neuron.local_batch_size_train,
        self.config.neuron.local_batch_size_train_effective,
        self.tokenizer,
        self.device,
    )

    self.scaler = torch.amp.GradScaler(enabled=True)

    if (self.local_progress.inner_step != 0) and ("." in revision):
        self.state_averager.reset_main_parameters(
            model_name,
            revision=".".join(
                revision.split(".")[:-1]
                + [str(get_min_local_inner_Step(self, model_name, epoch=epoch))]
            ),
        )

    bt.logging.debug(
        f"CPU Memory After Loading State {psutil.virtual_memory().available / 10**9} GB"
    )


def load_state_from_peer(
    self,
    repo_id=None,
    epoch=None,
    reload_inner_optimizer=True,
    reload_outer_optimizer=True,
    revision=None,
    use_fallback_model=True,
):
    try:
        state_loaded = False
        if epoch is None:
            self.global_progress.epoch = get_global_epoch(self)
            epoch = self.global_progress.epoch
        if repo_id is None:
            repo_id = self.config.neuron.global_model_name
        self.local_progress.inner_step = get_local_inner_step(
            self, repo_id, epoch=self.global_progress.epoch
        )

        bt.logging.debug("Model Weights Before Loading State")
        current_model_weights_sample = copy.copy(
            [layer for layer in self.model.parameters()][-2][-10:].tolist()
        )
        bt.logging.debug(current_model_weights_sample)

        bt.logging.debug(f"Old Model Tag: {self.local_progress.epoch}")

        if self.global_progress.epoch is not None:
            bt.logging.debug(
                f"Latest Model State Found On The HF Hub With The Tag: {self.global_progress.epoch}. Loading That Model State."
            )

            # Load model state with max retries
            MAX_ATTEMPTS = 3
            attempt = 0

            while attempt < MAX_ATTEMPTS:
                try:
                    load_model_optimizer_gradient_averager(
                        self,
                        local_model_name=repo_id,
                        epoch=epoch,
                        reload_inner_optimizer=reload_inner_optimizer,
                        reload_outer_optimizer=reload_outer_optimizer,
                        revision=revision,
                        use_fallback_model=use_fallback_model,
                    )
                    break

                except Exception as e:
                    attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        raise Exception(
                            f"Failed to load model after {MAX_ATTEMPTS} attempts: {str(e)}"
                        )
                    bt.logging.warning(
                        f"Failed to load model, retrying. Attempt {attempt}/{MAX_ATTEMPTS}. Error {str(e)}"
                    )

            state_loaded = True

            bt.logging.debug("Model Weights After Loading State")
            new_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-2][-10:].tolist()
            )
            bt.logging.debug(new_model_weights_sample)

            self.local_progress.epoch = epoch
            self.local_progress.samples_accumulated = 0
            bt.logging.debug(f"New Model Tag: {self.global_progress.epoch}")

            # # Clean up old cache
            # try:
            #     cleanup_old_cache(self, repo_id, revision)
            # except Exception as e:
            #     bt.logging.warning(f"Failed to cleanup cache: {str(e)}")

            # if repo_id != self.config.neuron.global_model_name:
            #     try:
            #         cleanup_old_cache(
            #             self,
            #             self.config.neuron.global_model_name,
            #             current_revision=None,
            #         )
            #     except Exception as e:
            #         bt.logging.warning(f"Failed to cleanup cache: {str(e)}")

        else:
            bt.logging.debug(f"Model With Tag: {epoch} Does Not Exist")

        return state_loaded

    except Exception as e:
        bt.logging.error(f"Error loading state: {str(e)}")
        return False


# TODO Remove this if score_bandwidth is deprecated
async def load_state_from_miner(self, peer, timeout: Optional[float] = None):
    metadata = None
    hivemind_logger.info(f"Downloading parameters from peer {peer}")
    try:
        stub = self.grad_averager.get_stub(
            self._p2p,
            peer,
            namespace=self.grad_averager.matchmaking_kwargs["prefix"],
        )
        stream = await stub.rpc_download_state_partial(averaging_pb2.DownloadRequest())
        current_tensor_parts, tensors = [], []

        # TODO merge this with hivemind.compression.deserialize_tensor_stream
        async for message in aiter_with_timeout(stream, timeout=timeout):
            if message.metadata:
                metadata = self.grad_averager.serializer.loads(message.metadata)
            if message.tensor_part.dtype and current_tensor_parts:
                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                tensors.append(
                    deserialize_torch_tensor(
                        combine_from_streaming(current_tensor_parts)
                    )
                )
                current_tensor_parts = []
            current_tensor_parts.append(message.tensor_part)
        if current_tensor_parts:
            tensors.append(
                deserialize_torch_tensor(combine_from_streaming(current_tensor_parts))
            )

        if not metadata:
            hivemind_logger.exception(f"Peer {peer} did not send its state")
            return

        hivemind_logger.info(f"Finished downloading state from {peer}")
        return metadata, tensors
    except Exception as e:
        hivemind_logger.exception(f"Failed to download state from {peer} - {repr(e)}")
        return None, None


def cleanup_old_cache(self, repo_id=None, current_revision=None):
    """Helper method to clean up old cache files"""
    if repo_id is None:
        repo_id = self.config.neuron.global_model_name
        current_revision = self.model.config._commit_hash

    cache_info = scan_cache_dir()
    broken_cache_list = [str(warning) for warning in cache_info.warnings]
    cache_dir = HF_HUB_CACHE
    cache_dir = Path(cache_dir).expanduser().resolve()
    bt.logging.info("Cache clearing warnings:")
    bt.logging.info(f"{cache_info.warnings}")

    # Delete cache using preferred huggingface cache clearing method
    if current_revision is None:
        for cache in cache_dir.iterdir():
            if repo_id.replace("/", "--") in str(cache):
                bt.logging.info(f"Deleting the entire cache folder for repo {repo_id}.")
                try:
                    shutil.rmtree(str(cache))
                except OSError as e:
                    bt.logging.info(
                        "Error: %s - %s deleting the entire cache folder for the repo: %s"
                        % (e.filename, e.strerror, repo_id)
                    )

    else:
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                revisions = sorted(
                    repo.revisions, key=lambda r: r.last_modified, reverse=True
                )

                bt.logging.info(
                    f"Found {len(revisions)} model revisions in .cache folder. Proceeding to delete all non-current revision."
                )
                for revision in revisions:
                    if (current_revision is not None) and (
                        revision.commit_hash == current_revision
                    ):
                        bt.logging.info(
                            f"Skipping cache for current revision {revision.commit_hash}"
                        )
                        continue
                    else:
                        bt.logging.info(
                            f"Deleting cache for revision {revision.commit_hash}"
                        )
                        cache_info.delete_revisions(revision.commit_hash).execute()
                break

    # Forcefully remove the entire cache folder for a model if it's corrupted
    if len(broken_cache_list) > 1:
        for cache in cache_dir.iterdir():
            if str(cache) in str(broken_cache_list):
                bt.logging.info(
                    f"Found repo {repo_id} in HF cache warning message. Proceeding to delete the entire cache folder."
                )
                try:
                    shutil.rmtree(str(cache))
                except OSError as e:
                    bt.logging.info(
                        "Error: %s - %s deleting the entire cache folder for the repo: %s"
                        % (e.filename, e.strerror, repo_id)
                    )


def upload_new_state(self, epoch: int, results: dict, block: int = None):
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            bt.logging.info(
                f"Pushing new model and optimizer state to HF Hub with tag {epoch}"
            )

            # Save and upload both model and optimizer state
            upload_success = save_and_upload_state(
                self, epoch=epoch, results=results, block=block
            )

            if upload_success:
                # Verify the upload
                updated_refs = list_repo_refs(
                    self.config.neuron.global_model_name,
                    repo_type="model",
                )
                new_tag = (
                    max(
                        [
                            int(tag.name.split(".")[1])
                            for tag in updated_refs.tags
                            if (
                                (len(tag.name.split(".")) == 3)
                                and (tag.name.split(".")[0] == __run__)
                            )
                        ]
                    )
                    if updated_refs.tags
                    else 0
                )
                bt.logging.info(f"Successfully pushed new model with tag {new_tag}")
                # Wait to allow out of sync miners to download new model state
                time.sleep(self.load_state_timeout)
                break

        except HfHubHTTPError as e:
            attempt += 1
            bt.logging.info(f"{e}. Loading State from Peer.")
            state_loaded = load_state_from_peer(self, epoch=self.global_progress.epoch)
            if state_loaded:
                break
        except Exception:
            attempt += 1
            bt.logging.warning(
                f"Failed To Upload Model To HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}."
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                bt.logging.error(
                    "Maximum Retry Limit Reached. Unable To Upload Model To HF Hub."
                )
                raise
    return upload_success


def save_and_upload_state(self, epoch: int, results: dict, block: int = None):
    """Unified function to save and upload both model and optimizer state"""
    batch_size = sum(
        [result for result in results["gathered"].values() if result is not None]
    )
    participating_peers = results["participating_peers"]
    failed_peers = results["failed_peers"]
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            with tempfile.TemporaryDirectory() as tmp_folder:
                bt.logging.info(
                    f"Preparing model and optimizer state for epoch {epoch}"
                )
                if block is not None:
                    self.model.config.last_allreduce_block = block
                self.model.config.inner_step = 0
                self.model.save_pretrained(tmp_folder)

                # Save outer optimizer state
                outer_optimizer_state = {
                    "optimizer_state_dict": self.state_averager.optimizer.state_dict(),
                    "learning_rate": self.state_averager.optimizer.param_groups[0][
                        "lr"
                    ],
                    "epoch": epoch,
                }
                torch.save(
                    outer_optimizer_state,
                    os.path.join(tmp_folder, "outer_optimizer.pt"),
                )

                # Save outer optimizer state
                inner_optimizer_state = {
                    "optimizer_state_dict": self.inner_optimizer.state_dict(),
                    "learning_rate": self.inner_optimizer.param_groups[0]["lr"],
                    "scheduler_state": self.scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    inner_optimizer_state,
                    os.path.join(tmp_folder, "inner_optimizer.pt"),
                )

                bt.logging.info(
                    f"Uploading model and optimizer states to repo: {self.config.neuron.global_model_name}"
                )

                # Upload everything in one go
                commit_message = f"Run {__run__}. Outer Step {epoch}. Inner Step {0}. Peers {len(participating_peers) - len(failed_peers)}."
                upload_folder(
                    folder_path=tmp_folder,
                    repo_id=self.config.neuron.global_model_name,
                    repo_type="model",
                    commit_message=commit_message,
                )

                # Create a tag for this version
                create_tag(
                    self.config.neuron.global_model_name,
                    repo_type="model",
                    tag=f"{__run__}.{epoch}.{0}",
                    tag_message=commit_message,
                )

                bt.logging.info(
                    f"Successfully pushed new model and optimizer state with tag {epoch} to repo: {self.config.neuron.global_model_name}"
                )
                return True

        except Exception as e:
            attempt += 1
            bt.logging.warning(
                f"Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                bt.logging.error(
                    "Maximum retry limit reached. Unable to upload state to HF Hub."
                )
                raise
    return False


def get_top_uid(self):
    all_reduce_scores_uids = [
        k
        for k, v in self.allreduce_status_dict.items()
        if (v == "SUCCESS")
        and (self.uid_tracker[int(k)]["model_huggingface_id"] is not None)
        and (
            requests.head(
                f"https://huggingface.co/api/models/{self.uid_tracker[int(k)]['model_huggingface_id']}"
            ).status_code
            == 200
        )
        and (
            (
                datetime.now(pytz.utc)
                - list_repo_commits(
                    self.uid_tracker[int(k)]["model_huggingface_id"], repo_type="model"
                )[0].created_at
            ).seconds
            < (60 * 60)
        )
    ]
    top_uid_list = [
        k
        for k, v in sorted(
            {
                u: self.metagraph.incentive[int(u)].item()
                for u in all_reduce_scores_uids
            }.items(),
            key=lambda item: item[1],
        )
    ]
    if top_uid_list != []:
        top_uid = top_uid_list[-1]
    bt.logging.info(f"Top UID Identified As {top_uid}")
    return top_uid
