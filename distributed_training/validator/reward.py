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

import asyncio
import copy
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

from huggingface_hub import hf_hub_download
# # GPU optimizations.
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Seeds
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

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
                n_pages=1,
                seed=uid,
            )
            # bt.logging.info(f"Fetched pages {pages} for UID {uid} at block {block}")
            random.seed(uid)
            random.shuffle(pages)

            dataset = await DatasetLoader.create(
                batch_size=self.local_batch_size_train,
                sequence_length=1024,
                pages_info=pages,
                tokenizer=self.tokenizer,
            )
            # bt.logging.info(f"Created dataset for UID {uid} at block {block}")

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


async def evaluate_model(
    self,
    model: torch.nn.Module,
    blocks: list[int],
    samples: list[int],
    uid: int,
    test_flag: bool = False,
) -> tuple[float, int]:
    model = model
    device = model.device
    total_loss = 0.0
    n_batches_total = 0
    n_batches_sampled = 0

    # TODO remove this after testing
    for block in blocks:
        
        dataset = await fetch_training_data(self, block, uid)
        # bt.logging.info(":pages: Fetched fineweb-edu pages")

        with torch.no_grad():
            model.eval()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                for i, batch in enumerate(dataset):
                    inputs, labels = batch
                    if inputs is None or len(inputs) == 0:
                        bt.logging.info(f"Empty batch at index {i}, skipping")
                        continue
                    n_batches_total += 1
                    # if i not in samples:
                    #     continue
                    # if test_flag:
                    #     breakpoint()
                    # bt.logging.info(f"Testing batch {i} for UID {uid}")
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(input_ids=inputs, labels=labels)
                        
                    total_loss += outputs.loss.item()
                    n_batches_sampled += 1
                    del inputs, labels, outputs
                    torch.cuda.empty_cache()

    return total_loss, n_batches_total, n_batches_sampled

async def score_uids(self, uids: list):
    """
    Score gradients for a group UIDs
    """
    blocks = [(5613899*2)]
    epoch = self.global_progress.epoch - 1
    test_time = time.time()
    
    model_huggingface_id = "distributed/llama-1b"
    if self.local_progress.epoch != epoch:
        load_state_from_peer(
            self,
            repo_id=model_huggingface_id,
            epoch=epoch,
            reload_inner_optimizer=True,
            reload_outer_optimizer=False,
            use_fallback_model=False,
        )
    
    for uid in uids:
        try:
            # self.uid_tracker[uid]["train/loss_abs"] = 0.5
            # self.uid_tracker[uid]["train/loss_rel"] = 0.5
            # continue
            # Sample dataset indicces to use for testing this uid
            bt.logging.info(f"UID: {uid}. Sampling dataset indices for testing")
            samples = random.sample([i for i in range(500)], 4)
            
            # Calculate loss before applying gradient
            bt.logging.info(f"UID: {uid}. Calculating loss before applying gradient")
            total_loss_before, n_batches_total_before, n_batches_sampled_before = await evaluate_model(self, model = self.model, blocks = blocks, samples = samples, uid=uid)
            average_loss_before = total_loss_before / n_batches_sampled_before
            self.uid_tracker[uid]["train/loss_before"] = average_loss_before
            bt.logging.info(f"UID {uid :03d}: Loss before: {average_loss_before}. Samples tested: {n_batches_sampled_before}/{n_batches_total_before}.")
            
            # Apply pseudo gradient for uid
            bt.logging.info(f"UID: {uid}. Applying pseudo gradient")
            model_t1 = copy.deepcopy(self.model)
            revision = f"5.{epoch}.{get_local_inner_step(self, repo_id=self.uid_tracker[uid]['train/model_huggingface_id'], epoch = epoch)}"
            self.uid_tracker[uid]["train/last_updated_revision"] = revision
            
            gradient = torch.load(
                hf_hub_download(
                    repo_id=self.uid_tracker[uid]['train/model_huggingface_id'],
                    filename="gradients.pt",
                    revision=revision,
                ),
                weights_only=True,
                map_location="cpu",
            )
            gradient.pop("metadata")
            bt.logging.info(f"UID {uid :03d}: Gradient loaded from {self.uid_tracker[uid]['train/model_huggingface_id']} with revision {revision}")
            self.update_model_with_gradient(model=model_t1, eval_uid=uid, eval_state_dict=gradient)
            bt.logging.info(f"UID {uid :03d}: Model updated with current gradient")
            # self.state_averager.optimizer.step()
            
            # Calculate loss after applying gradient
            total_loss_after, n_batches_total_after, n_batches_sampled_after = await evaluate_model(self, model = model_t1, blocks = blocks, samples=samples, uid = uid, test_flag=True)
            average_loss_after = total_loss_after / n_batches_sampled_after
            self.uid_tracker[uid]["train/loss_after"] = average_loss_after
            bt.logging.info(f"UID {uid :03d}: Loss after: {average_loss_after}. Samples tested: {n_batches_sampled_after}/{n_batches_total_after}.")
            
            # Calculate absolute and relative loss improvement
            if average_loss_before == 0:
                # Avoid division by zero
                self.uid_tracker[uid]["train/loss_abs"] = 0
                self.uid_tracker[uid]["train/loss_rel"] = 0
            else:
                self.uid_tracker[uid]["train/loss_abs"] = self.uid_tracker[uid]["train/loss_before"] - self.uid_tracker[uid]["train/loss_after"]
                self.uid_tracker[uid]["train/loss_rel"] = (self.uid_tracker[uid]["train/loss_before"] - self.uid_tracker[uid]["train/loss_after"])/self.uid_tracker[uid]["train/loss_before"]
            
            bt.logging.info(f"UID {uid :03d}: Absolute loss improvement: {self.uid_tracker[uid]['train/loss_abs']}")
            bt.logging.info(f"UID {uid :03d}: Relative loss improvement: {self.uid_tracker[uid]['train/loss_rel']}")
            
        except Exception as e:
            bt.logging.info(f"UID {uid :03d}: Error caclualting loss score: {e}")

        finally:
            # TODO: check gradient file cache
            self.uid_tracker[uid]["train/last_updated_time"] = test_time

    # # Delete temporary variables to free memory
    # del gradient
    # del model_t1
    # del total_loss_before, n_batches_total_before, n_batches_sampled_before
    # del total_loss_after, n_batches_total_after, n_batches_sampled_after
    # torch.cuda.empty_cache()

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
            self.uid_tracker[uid]["train/repo_valid_score"] = score_repo(
                self, self.uid_tracker[uid]["train/model_huggingface_id"]
            )
        except (RepositoryNotFoundError, RevisionNotFoundError, OSError) as e:
            # bt.logging.info(f"UID {uid} benchmarking failed with error {e}. Updating score to 0.")
            self.uid_tracker[uid]["train/repo_valid_score"] = False
        except Exception as e:
            bt.logging.info(
                f"UID {uid} benchmarking failed with error {e}. Keeping score as is."
            )
    bt.logging.info(
        {uid: self.uid_tracker[uid]["train/repo_valid_score"] for uid in self.uid_tracker}
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
        self.uid_tracker[uid].get("train/repo_valid_score", 0.0) for uid in self.uid_tracker
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
    self.report_train_scores()
