import copy
import gc
import os
import requests
import pytz
import shutil
import tempfile
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


def check_model_exists(repo_id: str, revision: Optional[str] = None) -> bool:
    try:
        if revision and revision != "None":
            list_repo_files(repo_id, revision=revision)
        else:
            list_repo_files(repo_id)
        return True
    except Exception as e:
        self.logger.info(f"Model or revision check failed with error: {e}")
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
    self.logger.debug(
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
        self.logger.info("Deleted State Averager and Gradient Averager")

    # Delete existing averag handler
    if hasattr(self, "avg_handler"):
        del self.avg_handler.model
        del self.avg_handler.inner_optimizer
        del self.avg_handler.grad_averager
        del self.avg_handler.state_averager
        del self.avg_handler
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Deleted Average Handler")

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
            self.logger.info(
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
                    self.logger.info("Deleted Inner Optimizer")

                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate_maximum,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                )
                self.logger.info(f"Loaded Inner Optimizer")

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
                self.logger.info(
                    f"Successfully Loaded Inner Optimizer State From {model_name} For Revision {revision}"
                )

                break

        except Exception as e:
            if model_name == model_name_list[-1]:
                raise Exception(f"Failed to load model despite repo existing: {str(e)}")
            else:
                self.logger.info(
                    f"Failed to load model despite repo existing: {str(e)}"
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

    # Set outer optimizer
    self.outer_optimizer = partial(torch.optim.SGD, lr=1, momentum=0.9, nesterov=True)

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
    self.logger.info("Successfully Loaded Gradient Averager")

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
    self.logger.info("Successfully Loaded State Averager")

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

            self.logger.info(
                f"Successfully Loaded Outer Optimizer State From {global_model_name} For Revision {'.'.join(revision.split('.')[:-1] + ['0'])}"
            )

        except Exception as e:
            self.logger.warning(
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

    self.logger.debug(
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

        self.logger.debug("Model Weights Before Loading State")
        current_model_weights_sample = copy.copy(
            [layer for layer in self.model.parameters()][-2][-10:].tolist()
        )
        self.logger.debug(current_model_weights_sample)

        self.logger.debug(f"Old Model Tag: {self.local_progress.epoch}")

        if self.global_progress.epoch is not None:
            self.logger.debug(
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
                    self.logger.warning(
                        f"Failed to load model, retrying. Attempt {attempt}/{MAX_ATTEMPTS}. Error {str(e)}"
                    )

            state_loaded = True

            self.logger.debug("Model Weights After Loading State")
            new_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-2][-10:].tolist()
            )
            self.logger.debug(new_model_weights_sample)

            self.local_progress.epoch = epoch
            self.local_progress.samples_accumulated = 0
            self.logger.debug(f"New Model Tag: {self.global_progress.epoch}")

            # # Clean up old cache
            # try:
            #     cleanup_old_cache(self, repo_id, revision)
            # except Exception as e:
            #     self.logger.warning(f"Failed to cleanup cache: {str(e)}")

            # if repo_id != self.config.neuron.global_model_name:
            #     try:
            #         cleanup_old_cache(
            #             self,
            #             self.config.neuron.global_model_name,
            #             current_revision=None,
            #         )
            #     except Exception as e:
            #         self.logger.warning(f"Failed to cleanup cache: {str(e)}")

        else:
            self.logger.debug(f"Model With Tag: {epoch} Does Not Exist")

        return state_loaded

    except Exception as e:
        self.logger.error(f"Error loading state: {str(e)}")
        return False


def cleanup_old_cache(self, repo_id=None, current_revision=None):
    """Helper method to clean up old cache files"""
    if repo_id is None:
        repo_id = self.config.neuron.global_model_name
        current_revision = self.model.config._commit_hash

    cache_info = scan_cache_dir()
    broken_cache_list = [str(warning) for warning in cache_info.warnings]
    cache_dir = HF_HUB_CACHE
    cache_dir = Path(cache_dir).expanduser().resolve()
    self.logger.info("Cache clearing warnings:")
    self.logger.info(f"{cache_info.warnings}")

    # Delete cache using preferred huggingface cache clearing method
    if current_revision is None:
        for cache in cache_dir.iterdir():
            if repo_id.replace("/", "--") in str(cache):
                self.logger.info(
                    f"Deleting the entire cache folder for repo {repo_id}."
                )
                try:
                    shutil.rmtree(str(cache))
                except OSError as e:
                    self.logger.info(
                        "Error: %s - %s deleting the entire cache folder for the repo: %s"
                        % (e.filename, e.strerror, repo_id)
                    )

    else:
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                revisions = sorted(
                    repo.revisions, key=lambda r: r.last_modified, reverse=True
                )

                self.logger.info(
                    f"Found {len(revisions)} model revisions in .cache folder. Proceeding to delete all non-current revision."
                )
                for revision in revisions:
                    if (current_revision is not None) and (
                        revision.commit_hash == current_revision
                    ):
                        self.logger.info(
                            f"Skipping cache for current revision {revision.commit_hash}"
                        )
                        continue
                    else:
                        self.logger.info(
                            f"Deleting cache for revision {revision.commit_hash}"
                        )
                        cache_info.delete_revisions(revision.commit_hash).execute()
                break

    # Forcefully remove the entire cache folder for a model if it's corrupted
    if len(broken_cache_list) > 1:
        for cache in cache_dir.iterdir():
            if str(cache) in str(broken_cache_list):
                self.logger.info(
                    f"Found repo {repo_id} in HF cache warning message. Proceeding to delete the entire cache folder."
                )
                try:
                    shutil.rmtree(str(cache))
                except OSError as e:
                    self.logger.info(
                        "Error: %s - %s deleting the entire cache folder for the repo: %s"
                        % (e.filename, e.strerror, repo_id)
                    )


def upload_new_state(self, epoch: int, results: dict, block: int = None):
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            self.logger.info(
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
                self.logger.info(f"Successfully pushed new model with tag {new_tag}")
                # Wait to allow out of sync miners to download new model state
                time.sleep(self.load_state_timeout)
                break

        except HfHubHTTPError as e:
            attempt += 1
            self.logger.info(f"{e}. Loading State from Peer.")
            state_loaded = load_state_from_peer(self, epoch=self.global_progress.epoch)
            if state_loaded:
                break
        except Exception:
            attempt += 1
            self.logger.warning(
                f"Failed To Upload Model To HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}."
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                self.logger.error(
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
                self.logger.info(
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

                self.logger.info(
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

                self.logger.info(
                    f"Successfully pushed new model and optimizer state with tag {epoch} to repo: {self.config.neuron.global_model_name}"
                )
                return True

        except Exception as e:
            attempt += 1
            self.logger.warning(
                f"Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                self.logger.error(
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
    self.logger.info(f"Top UID Identified As {top_uid}")
    return top_uid
