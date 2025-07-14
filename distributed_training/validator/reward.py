# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 KMFODA

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
import json
import math
import random
import time
from datetime import datetime

import base58
import bittensor as bt
import numpy as np
import pytz
import torch
import torch.nn.functional as F
from hivemind.p2p import PeerID
from huggingface_hub import list_repo_commits, HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from transformers import AutoConfig, AutoModelForCausalLM

from distributed_training import __run__
from distributed_training.data.dataset import DatasetLoader
from distributed_training.utils.progress_tracker import (
    get_local_epoch,
    get_local_inner_step,
)
from distributed_training.utils.state_loader import (
    check_model_exists,
    cleanup_old_cache,
    load_state_from_peer,
)

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set scoring weights
TRAIN_SCORE_WEIGHT = 0.75
ALL_REDUCE_SCORE_WEIGHT = 0.25

api = HfApi()


async def score_blacklist(self, uids):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        if self.uids_to_peerids[uid][0] == None:
            scores[i] = 0.0
        elif self.uids_to_peerids[uid][0] in self.run_peer_id_list:
            scores[i] = 1.0
        else:
            scores[i] = 0.0

    return scores


async def score_bandwidth(self, uids, timeout=30):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer_id = self.uids_to_peerids[uid][0]

        if peer_id is None:
            peer = None
        else:
            peer = PeerID(base58.b58decode(peer_id))

        if peer is None:
            scores[i] = 0

        else:
            try:
                start_time = time.perf_counter()

                metadata, tensors = await asyncio.wait_for(
                    self.load_state_from_miner(peer), timeout=timeout
                )
                end_time = time.perf_counter()

                if (metadata is None) or (tensors is None):
                    scores[i] = 0
                else:
                    scores[i] = 1 - ((end_time - start_time) / timeout)

                bt.logging.info(f"Reward for peer {peer} is {scores[i]}")

            except Exception as e:
                bt.logging.info(f"Failed to download state from {peer} - {repr(e)}")
                scores[i] = 0
                bt.logging.info(f"Reward for peer {peer} is {scores[i]}")

    return scores


def score_failed_senders(self, uids, failed_peers, participating_peers):
    scores = torch.FloatTensor([0.0 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer_id = self.uids_to_peerids.get(uid)[0]

        if peer_id in participating_peers:
            if peer_id in failed_peers:
                bt.logging.info(f"UID:{uid} - Failed participating peer")
                scores[i] = 0.0
            else:
                bt.logging.info(f"UID:{uid} - Successful participating peer")
                scores[i] = 1.0
        else:
            bt.logging.info(f"UID:{uid} - Non participating peer")
            scores[i] = 0.0

    return scores


async def fetch_training_data(self, block, uid):
    """Async function to fetch training data"""
    attempt = 0
    while attempt < self.retry_limit:
        try:
            pages = await DatasetLoader.next_pages(
                offset=block,
                n_pages=35,
                seed=uid,
            )
            random.seed(uid)
            random.shuffle(pages)

            dataset = await DatasetLoader.create(
                batch_size=self.local_batch_size_train,
                sequence_length=1024,
                pages_info=pages,
                tokenizer=self.tokenizer,
            )

            return dataset
        except Exception as e:
            bt.logging.error(f"Error fetching training data: {str(e)}")
            attempt += 1
            bt.logging.warning(
                f"Failed to fetch data, retrying. Attempt {attempt}/{self.retry_limit}"
            )
            if attempt < self.retry_limit:
                time.sleep(self.retry_delay * attempt)  # Wait before the next retry
            else:
                bt.logging.error("Maximum retry limit reached. Unable to fetch data.")
                raise


async def score_uid(self, uid: int):
    """Score a single UID"""
    target_blocks = self.config.neuron.target_n_blocks
    latest_commit = None
    model_huggingface_id = self.uid_tracker[uid]["model_huggingface_id"]
    local_epoch = get_local_epoch(self, model_huggingface_id)
    self.local_progress.inner_step = get_local_inner_step(self, model_huggingface_id)
    revision = f"{__run__}.{local_epoch}.{self.local_progress.inner_step}"

    blocks = []
    time_delta = 0
    try:
        if model_huggingface_id is None:
            scores = 0
            raise Exception(f"Score 0 for UID {uid}: HuggingFace Repo Id is None")
        elif not check_model_exists(model_huggingface_id, revision):
            scores = 0
            raise Exception(
                f"Score 0 for UID {uid}: HuggingFace Repo Id {self.uid_tracker[uid]['model_huggingface_id']} Doesn't Exist"
            )
        elif (local_epoch is None) or (local_epoch != self.global_progress.epoch):
            scores = 0
            raise Exception(
                f"Score 0 for UID {uid}: Local Epoch {local_epoch} != Global Epoch {self.global_progress.epoch}"
            )

        cleanup_old_cache(
            self,
            repo_id=model_huggingface_id,
            current_revision=None,
        )

        commits = list_repo_commits(model_huggingface_id, repo_type="model")[:2]
        latest_commit = commits[0].commit_id
        previous_commit = commits[1].commit_id
        time_delta = (commits[0].created_at - commits[1].created_at).seconds

        load_state_from_peer(
            self,
            repo_id=model_huggingface_id,
            epoch=local_epoch,
            reload_inner_optimizer=True,
            reload_outer_optimizer=False,
            revision=previous_commit,
            use_fallback_model=False,
        )
        # Only set self.local_progress.epoch if model is correct format
        self.local_progress.epoch = get_local_epoch(self, model_huggingface_id)
        self.local_progress.samples_accumulated = 0
        inner_step_t0 = (
            self.model.config.inner_step
            if "inner_step" in self.model.config.__dict__
            else 0
        )
        self.local_progress.inner_step = inner_step_t0

        model_final = AutoModelForCausalLM.from_pretrained(
            model_huggingface_id, revision=latest_commit, trust_remote_code=True
        )
        inner_step_t1 = (
            model_final.config.inner_step
            if "inner_step" in model_final.config.__dict__
            else 0
        )

        if time_delta < (30 * target_blocks):
            scores = 0
            bt.logging.info(
                f"Score 0 for UID {uid}: Time Delta {time_delta} > 30 * Target Blocks {target_blocks}"
            )
        elif (sum(p.numel() for p in model_final.parameters()) > 1100048384) or (
            sum(p.numel() for p in self.model.parameters()) > 1100048384
        ):
            scores = 0
            bt.logging.info(
                f"Score 0 for UID {uid}: Repo {model_huggingface_id} Failed Model Size Validation"
            )
        elif ("block_list" in self.model.config.__dict__) and (
            len(self.model.config.block_list) > target_blocks
        ):
            scores = 0
            bt.logging.info(
                f"Score 0 for UID {uid}: Block List Length {len(self.model.config.block_list)} > Target Blocks {target_blocks}"
            )
        elif inner_step_t0 >= inner_step_t1:
            scores = 0
            bt.logging.info(
                f"Score 0 for UID {uid}: Inner Step T0 {inner_step_t0} == Inner Step T1 {inner_step_t1}"
            )
        else:
            blocks = model_final.config.block_list
            self.running_loss = 0.0
            self.batch_count = 0
            self.local_progress.samples_accumulated = 0
            for block in blocks:
                bt.logging.debug(":pages: Fetching fineweb-edu pages")
                dataset = await fetch_training_data(self, block, uid)
                for inputs, labels in dataset:
                    # Move to device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(input_ids=inputs, labels=labels)
                        loss = outputs.loss / self.number_of_local_steps

                    if math.isnan(loss.item()):
                        raise Exception(f"Score 0 for UID {uid}: NaN detected in Loss")

                    loss.backward()

                    self.running_loss += loss.item() * self.number_of_local_steps
                    self.batch_count += 1
                    self.local_progress.loss = self.running_loss / self.batch_count

                    self.local_progress.samples_accumulated += (
                        self.local_batch_size_train
                    )

                    # Check if we've accumulated enough samples for a step
                    if (
                        self.local_progress.samples_accumulated
                        >= self.local_batch_size_train_effective
                    ):
                        bt.logging.info(
                            f":training:  Outer Step: {self.local_progress.epoch} | "
                            f"Inner Step: {self.local_progress.inner_step} | "
                            f"Learning Rate: {self.inner_optimizer.param_groups[0]['lr']:.8f} | "
                            f"Average Loss: {self.local_progress.loss:.2f}"
                        )

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.inner_optimizer.step()

                        self.scheduler.step()

                        self.inner_optimizer.zero_grad()

                        self.local_progress.inner_step += 1

                        self.running_loss = 0.0
                        self.batch_count = 0

                        self.local_progress.samples_accumulated = 0

            scores = score_models(self.model, model_final)
            if self.local_progress.inner_step == inner_step_t0:
                scores = 0
    except Exception as e:
        # TODO if OSError and e.rrno == errno.ENOSPC score as 1
        scores = 0.0
        bt.logging.info(
            f"Score {int(scores)} for UID {uid}: Forward Loop Failed With Error: {e}"
        )

    finally:
        if model_huggingface_id is not None:
            cleanup_old_cache(
                self,
                repo_id=model_huggingface_id,
                current_revision=None,
            )

    self.uid_tracker[uid]["last_commit"] = latest_commit
    self.uid_tracker[uid]["train_number_of_blocks"] = len(blocks)
    self.uid_tracker[uid]["train_duration"] = time_delta
    self.uid_tracker[uid]["train_similarity_score_last_updated"] = time.time()
    self.uid_tracker[uid]["train_similarity_score"] = (
        0 if math.isnan(scores) else scores
    )
    self.uid_tracker[uid]["train_validation_count"] += 1
    self.uid_tracker[uid]["loss"] = self.local_progress.loss

    bt.logging.info(f"UID {uid} Current Train Score: {scores}")
    update_individual_scores(self, uid, target_blocks)


def update_individual_scores(self, uid, target_blocks):
    uid_data = self.uid_tracker[uid]

    if uid_data.get("train_duration", 0) != 0:
        uid_data["train_score"] = (
            uid_data["train_similarity_score"]
            * min(uid_data["train_number_of_blocks"], target_blocks)
        ) / uid_data["train_duration"]
    else:
        uid_data["train_score"] = 0.0


def score_models(model_1, model_2):
    """Calculate the cosine similarity score between two model states"""
    score = 0
    index = 0

    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        score += (
            F.cosine_similarity(param_1.to("cpu"), param_2.to("cpu"), dim=0)
            .mean()
            .item()
        )
        index += 1

    average_score = score / index
    return average_score


def score_repo(self, repo_id: str) -> bool:
    local_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    if (
        (self.global_model_config.hidden_size != local_config.hidden_size)
        or (
            self.global_model_config.num_attention_heads
            != local_config.num_attention_heads
        )
        or (
            self.global_model_config.num_hidden_layers != local_config.num_hidden_layers
        )
        or (
            self.global_model_config.num_key_value_heads
            != local_config.num_key_value_heads
        )
    ):
        return False
    latest_commit = api.repo_info(repo_id).lastModified

    if (datetime.now(pytz.utc) - latest_commit).seconds > (
        self.config.neuron.target_n_blocks * 60 * 10
    ):
        return False
    return True


def benchmark_uids(self):
    for uid in self.uid_tracker:
        try:
            self.uid_tracker[uid]["repo_valid_score"] = score_repo(
                self, self.uid_tracker[uid]["model_huggingface_id"]
            )
        except (RepositoryNotFoundError, RevisionNotFoundError, OSError) as e:
            # bt.logging.info(f"UID {uid} benchmarking failed with error {e}. Updating score to 0.")
            self.uid_tracker[uid]["repo_valid_score"] = False
        except Exception as e:
            breakpoint()
            bt.logging.info(
                f"UID {uid} benchmarking failed with error {e}. Keeping score as is."
            )
    bt.logging.info(
        {uid: self.uid_tracker[uid]["repo_valid_score"] for uid in self.uid_tracker}
    )


def update_all_reduce_scores(self):
    try:
        if self.allreduce_status_dict != {}:
            for uid in self.allreduce_status_dict.keys():
                if (self.allreduce_status_dict[uid] == "SUCCESS") or (
                    self.allreduce_status_dict[uid] == "NON_PARTICIPATING"
                ):
                    score = 1
                else:
                    score = 0
                if self.uid_tracker[int(uid)]["all_reduce_score"] != score:
                    self.uid_tracker[int(uid)]["all_reduce_count"] += 1
                self.uid_tracker[int(uid)]["all_reduce_score"] = score
    except Exception as e:
        bt.logging.info(f"Error {e} updating all_reduce scores")




async def is_dataset_server_responsive(self) -> bool:
    """
    Performs a single health check to see if the Hugging Face dataset server is responsive.

    Returns:
        bool: True if the server is accessible, False otherwise.
    """
    health_check_retry_limit = 5
    health_check_retry_delay = 15 

    bt.logging.info("Performing health check on Hugging Face dataset server...")
    for attempt in range(health_check_retry_limit):
        try:
            await DatasetLoader.next_pages(offset=1, n_pages=1, seed=1)
            bt.logging.success("Hugging Face dataset server is responsive.")
            return True
        except Exception as e:
            bt.logging.warning(
                f"Health check failed (Attempt {attempt + 1}/{health_check_retry_limit}). Error: {e}"
            )
            if attempt < health_check_retry_limit - 1:
                await asyncio.sleep(health_check_retry_delay)

    bt.logging.error(
        f"Health check failed after {health_check_retry_limit} attempts. The dataset server appears to be down."
    )
    return False

def update_total_scores(self):
    # Update AllReduce stats from the latest round
    update_all_reduce_scores(self)

    # Sort uid tracker
    self.uid_tracker = dict(sorted(self.uid_tracker.items()))

    # Normalise each type of reward
    train_scores = [
        self.uid_tracker[uid].get("train_score", 0.0) for uid in self.uid_tracker
    ]
    all_reduce_scores = [
        self.uid_tracker[uid].get("all_reduce_score", 0.0) for uid in self.uid_tracker
    ]
    repo_valid_scores = [
        self.uid_tracker[uid].get("repo_valid_score", 0.0) for uid in self.uid_tracker
    ]

    train_scores_normalised = (
        np.linalg.norm(train_scores, ord=1, axis=0, keepdims=True)
        if any(train_scores)
        else np.array(1.0)
    ).item()
    all_reduce_scores_normalised = (
        np.linalg.norm(all_reduce_scores, ord=1, axis=0, keepdims=True)
        if any(all_reduce_scores)
        else np.array(1.0)
    ).item()
    repo_valid_scores_normalised = (
        np.linalg.norm(repo_valid_scores, ord=1, axis=0, keepdims=True)
        if any(repo_valid_scores)
        else np.array(1.0)
    ).item()

    # Catch 0 and NaN norms to avoid division by zero
    if (train_scores_normalised == 0) or np.isnan(train_scores_normalised):
        train_scores_normalised = 1.0
    if all_reduce_scores_normalised == 0 or np.isnan(all_reduce_scores_normalised):
        all_reduce_scores_normalised = 1.0
    if repo_valid_scores_normalised == 0 or np.isnan(repo_valid_scores_normalised):
        repo_valid_scores_normalised = 1.0

    # Update total scores with repo_valid_score if train_score or all_reduce_score are 0
    # Otherwise score using weighted train_score and all_reduce_score
    for uid_key in self.uid_tracker:
        uid_data = self.uid_tracker[uid_key]
        train_score = uid_data.get("train_score", 0.0)
        all_reduce_score = uid_data.get("all_reduce_score", 0.0)
        repo_valid_score = uid_data.get("repo_valid_score", 0.0)

        if (uid_data["train_validation_count"] == 0) and (
            uid_data["all_reduce_count"] == 0
        ):
            normalized_repo_valid_score = (
                repo_valid_score / repo_valid_scores_normalised
            )
            uid_data["total_score"] = normalized_repo_valid_score
        else:
            normalized_train_score = (
                TRAIN_SCORE_WEIGHT * train_score
            ) / train_scores_normalised
            normalized_all_reduce_score = (
                ALL_REDUCE_SCORE_WEIGHT * all_reduce_score
            ) / all_reduce_scores_normalised
            uid_data["total_score"] = (
                normalized_train_score + normalized_all_reduce_score
            ) * repo_valid_score

    # Add metrics reporting
    self.report_scoring_metrics()
