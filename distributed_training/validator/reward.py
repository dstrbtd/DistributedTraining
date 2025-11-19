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
import os
import json
from io import StringIO
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
from distributed_training.utils.progress_tracker import get_progress, get_r2_client
from distributed_training.utils.state_loader import (
    cleanup_old_cache,
    load_state_from_peer,
)
from distributed_training.utils.r2 import r2_download
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
    set_model_state_dict,
)

from rich.console import Console
from rich.table import Table

import torch.distributed as dist
import email.utils
from datetime import datetime, timezone
from botocore.session import get_session

# Set scoring weights
TRAIN_SCORE_WEIGHT = 0.75
ALL_REDUCE_SCORE_WEIGHT = 0.25
MAX_UPLOAD_INTERVAL = 2400  # Seconds

api = HfApi()


async def fetch_training_data(
    self, block: int, uid: int, n_pages: int
) -> DatasetLoader:
    """
    Async function to fetch training data

    Args:
        block (_type_): the block number used to seed the data fetching.
        uid (_type_): the uid of the miner to fetch data for.
        n_pages (int): number of pages to fetch.

    Returns:
        DatasetLoader: An instance of DatasetLoader containing the training data.

    """
    attempt = 0
    while attempt < self.retry_limit:
        try:
            pages = await DatasetLoader.next_pages(
                offset=block,
                n_pages=n_pages,
                seed=uid + self.local_rank,
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
            dist.all_reduce(dataset_length, op=dist.ReduceOp.MIN, group=self.gloo_group)
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
                self.logger.error("Maximum retry limit reached. Unable to fetch data.")
                raise


async def evaluate_model(
    self,
    model: torch.nn.Module,
    blocks: list[int],
    uid: int,
    n_pages: int,
    samples: list[int] = None,
) -> tuple[float, int, int]:
    """
    Evaluate the model on the training data for a given UID.

    Args:
        model (torch.nn.Module): The model to evaluate.
        blocks (list[int]): List of block numbers to use for fetching data.
        uid (int): The UID of the miner to evaluate.
        n_pages (int): Number of pages to fetch for evaluation.
        samples (list[int], optional): Sample indices to use for testing. Defaults to None.
        test_flag (bool, optional): Flag to indicate if this is a test run. Defaults to False.

    Returns:
        tuple[float, int]: Total loss, number of batches processed, and number of batches sampled.
    """
    model = model
    device = model.device
    total_loss = 0.0
    n_batches_total = 0
    n_batches_sampled = 0

    for block in blocks:
        dataset = await fetch_training_data(self, block, uid, n_pages)

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dataset):
                inputs, labels = batch
                if inputs is None or len(inputs) == 0:
                    self.logger.info(f"Empty batch at index {i}, skipping")
                    continue
                n_batches_total += 1
                if (samples is not None) and (i not in samples):
                    continue
                inputs = inputs.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=inputs, labels=inputs)

                total_loss += outputs.loss.item()
                n_batches_sampled += 1
                del inputs, labels, outputs
                torch.cuda.empty_cache()

    return total_loss, n_batches_total, n_batches_sampled


async def evaluate_with_gradient(self, uid, model_base, blocks, revision, prefix):
    """
    Apply pseudo gradient for a UID and evaluate loss before and after.

    Args:
        uid (int): UID being evaluated.
        model_base (torch.nn.Module): The model to copy before applying gradient.
        blocks (list[int]): Data blocks to evaluate on.
        revision (str): Gradient file revision identifier.

    Returns:
        tuple: (average_loss_before, average_loss_after)
    """
    # 1. Evaluate loss before applying gradient
    self.logger.info(f"UID {uid:03d}: Calculating loss before applying gradient")
    (
        total_loss_before,
        n_batches_total_before,
        n_batches_sampled_before,
    ) = await evaluate_model(
        self, model=model_base, blocks=blocks, uid=uid, samples=None, n_pages=1
    )

    # local aggregates
    total_loss_before_tensor = torch.tensor(
        [total_loss_before], dtype=torch.float64, device="cuda"
    )
    n_batches_sampled_before_tensor = torch.tensor(
        [n_batches_sampled_before], dtype=torch.int64, device="cuda"
    )

    # sum across all ranks (in-place; now identical on every rank)
    dist.all_reduce(total_loss_before_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_batches_sampled_before_tensor, op=dist.ReduceOp.SUM)

    total_loss_before = float(total_loss_before_tensor.item())
    n_batches_sampled_before = int(n_batches_sampled_before_tensor.item())

    average_loss_before = total_loss_before / n_batches_sampled_before

    self.logger.info(
        f"UID {uid:03d}: Model loss before gradient update {average_loss_before:6f}"
    )

    # 2. Load and apply pseudo gradient
    self.logger.info(f"UID {uid:03d}: Applying pseudo gradient")

    model_t1 = model_base

    r2 = get_r2_client(self, uid, donwload_on_all_ranks=True)
    self.logger.info(f"UID {uid:03d}: Got r2 client")
    gradient_path = r2_download(
        self,
        r2=r2,
        bucket=f"{self.config.neuron.global_model_name.split('/')[-1]}-{uid:03d}",
        key=f"{prefix}gradients.pt",
        donwload_on_all_ranks=False,
        run_on_all_ranks=True,
        destination=os.path.join(
            os.getcwd(),
            f"{self.config.neuron.global_model_name.split('/')[-1]}-{uid:03d}",
        ),
    )
    gradient = torch.load(
        gradient_path,
        weights_only=True,
        map_location="cpu",
    )
    gradient.pop("metadata")

    self.logger.info(f"UID {uid:03d}: Gradient loaded with revision {revision}")
    self.update_model_with_pseudo_gradient(model=model_t1, uid=uid, state_dict=gradient)
    self.logger.info(f"UID {uid:03d}: Model updated with current gradient")

    # 3. Evaluate loss after applying gradient
    (
        total_loss_after,
        n_batches_total_after,
        n_batches_sampled_after,
    ) = await evaluate_model(
        self, model=model_t1, blocks=blocks, uid=uid, samples=None, n_pages=1
    )

    # local aggregates
    total_loss_after_tensor = torch.tensor(
        [total_loss_after], dtype=torch.float64, device="cuda"
    )
    n_batches_sampled_after_tensor = torch.tensor(
        [n_batches_sampled_after], dtype=torch.int64, device="cuda"
    )

    # sum across all ranks (in-place; now identical on every rank)
    dist.all_reduce(total_loss_after_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_batches_sampled_after_tensor, op=dist.ReduceOp.SUM)

    total_loss_after = float(total_loss_after_tensor.item())
    n_batches_sampled_after = int(n_batches_sampled_after_tensor.item())

    average_loss_after = total_loss_after / n_batches_sampled_after

    self.logger.info(
        f"UID {uid:03d}: Model loss after gradient update {average_loss_after:6f}"
    )

    # Cleanup variables taking up memory
    del gradient
    del model_t1
    del total_loss_before, n_batches_total_before, n_batches_sampled_before
    del total_loss_after, n_batches_total_after, n_batches_sampled_after
    torch.cuda.empty_cache()

    # Remove repo from cache
    cleanup_old_cache(self)

    return average_loss_before, average_loss_after


def compute_loss_improvement(before: float, after: float) -> dict:
    """Compute the absolute and relative loss improvement."""
    absolute = before - after
    relative = 0 if before == 0 else absolute / before
    return {
        "before": before,
        "after": after,
        "absolute": absolute,
        "relative": relative,
    }


def get_uids_blocks(self, uid: int, prefix=str) -> list[int]:
    """"""
    bucket_name = f"{self.config.neuron.global_model_name.split('/')[-1]}-{uid:03d}"
    r2 = get_r2_client(self, uid, donwload_on_all_ranks=True)
    config_path = r2_download(
        self,
        r2=r2,
        bucket=bucket_name,
        key=f"{prefix}config.json",
        donwload_on_all_ranks=True,
        run_on_all_ranks=True,
        destination=bucket_name,
    )
    uid_blocks = json.load(open(config_path))["block_list"]
    # if False:
    if (self.current_block - max(uid_blocks)) > (
        self.config.neuron.blocks_per_allreduce / 2
    ):
        raise Exception(
            f"Uploaded datatset block older than {((self.config.neuron.blocks_per_allreduce / 2) *12) / 60} minutes"
        )
    else:
        random.seed(uid)
        assgined_blocks = random.sample(uid_blocks, 1)
        return assgined_blocks


def reset_uid_train_scores(self, uid: int):
    """Penalize a uid by resetting thier train.random and train.assigned scores"""
    random_train_scores = {
        "before": 0.0,
        "after": 0.0,
        "absolute": 0.0,
        "relative": 0.0,
    }
    assigned_train_score = {
        "before": 0.0,
        "after": 0.0,
        "absolute": 0.0,
        "relative": 0.0,
    }
    for k, v in random_train_scores.items():
        setattr(self.uid_tracker[uid].train.random, k, v)
    for k, v in assigned_train_score.items():
        setattr(self.uid_tracker[uid].train.assigned, k, v)


async def score_uids(self, epoch: int, uids: list):
    """
    Score each UID by calculating the loss before and after applying their pseudo gradient.

    This scoring process is inspired by Templar's Gauntlet [1], particularly the approach for comparing UIDs'
    loss using random and assigned datapoints pre and post applying their shared pseudo gradients.

    References:
        [1] Incentivizing Permissionless Distributed Learning of LLMs
            ArXiv: https://arxiv.org/abs/2505.21684
            Code: https://github.com/tplr-ai/templar

    Args:
        uids (list): UIDs of miners to be evaluated.
    """
    if (
        self.blocks_since_allreduce < (self.config.neuron.blocks_per_allreduce / 2)
    ) and (self.global_progress.epoch != 0):
        prefix = f"epoch-{epoch}/"
    else:
        prefix = ""
    prefix = f"epoch-{epoch}/"

    test_time = time.time()

    if self.master:
        # Sample a random evaluation block
        random_blocks = random.sample(
            range(self.current_block + 20, self.current_block * 10), 1
        )
    else:
        random_blocks = [self.current_block]
    dist.broadcast_object_list(random_blocks, src=0, group=self.gloo_group)
    self.logger.info(random_blocks)

    # Get revisions for all UIDs
    if self.master:
        for uid in uids:
            self.uid_tracker[
                uid
            ].train.revision = f"{__run__}.{epoch}.{get_progress(self, 'local', uid=uid, donwload_on_all_ranks=False)[1]}"

    # Sync global model if behind
    if self.local_progress.epoch != epoch:
        load_state_from_peer(
            self,
            uid=self.master_uid,
            epoch=epoch,
            reload_inner_optimizer=True,
            reload_outer_optimizer=False,
            use_fallback_model=False,
        )

    # Extract model state for future resets
    model_state_dict = get_model_state_dict(
        self.model,
        options=StateDictOptions(
            full_state_dict=False,  # Only save local shards, not full model
            cpu_offload=True,  # Automatically offload to CPU
        ),
    )
    for uid in uids:
        score_status = torch.tensor([0])
        try:
            # Reset model state dict before an evaluation
            set_model_state_dict(
                self.model,
                dict(model_state_dict),
                options=StateDictOptions(
                    full_state_dict=False,  # We saved only local shards, so load them back as such
                    strict=True,  # Ensure all keys match
                ),
            )

            if self.master:
                revision_list = [self.uid_tracker[uid].train.revision]
            else:
                revision_list = [""]
            dist.broadcast_object_list(revision_list, src=0, group=self.gloo_group)
            revision = revision_list[0]

            # Revision Checks
            if int(revision.split(".")[-1]) == 0:
                raise Exception(f"Revision {revision} has 0 inner steps")

            # ──────────────────────────────────────────────────────────────────────────
            # Step 1: Evaluate on random unseen data
            # ──────────────────────────────────────────────────────────────────────────

            loss_scores = compute_loss_improvement(
                *await evaluate_with_gradient(
                    self=self,
                    uid=uid,
                    model_base=self.model,
                    blocks=random_blocks,
                    revision=revision,
                    prefix=prefix,
                )
            )

            if self.master:
                for k, v in loss_scores.items():
                    setattr(self.uid_tracker[uid].train.random, k, v)

            self.logger.info(
                f"UID {uid:03d}: Random <=> Absolute loss improvement: {loss_scores['absolute']:.6f}"
            )
            self.logger.info(
                f"UID {uid:03d}: Random <=> Relative loss improvement: {loss_scores['relative']:.6f}"
            )

            # ──────────────────────────────────────────────────────────────────────────
            # Step 2: Evaluate on UID's assigned data
            # ──────────────────────────────────────────────────────────────────────────

            # Reset model state dict before an evaluation
            set_model_state_dict(
                self.model,
                dict(model_state_dict),
                options=StateDictOptions(
                    full_state_dict=False,  # We saved only local shards, so load them back as such
                    strict=True,  # Ensure all keys match
                ),
            )

            self.logger.info(f"UID {uid:03d}: Sampling dataset indices for testing")
            self.set_current_block_across_ranks()
            assigned_block = get_uids_blocks(self, uid, prefix)
            self.logger.info(assigned_block)

            loss_scores = compute_loss_improvement(
                *await evaluate_with_gradient(
                    self=self,
                    uid=uid,
                    model_base=self.model,
                    blocks=assigned_block,
                    revision=revision,
                    prefix=prefix,
                )
            )

            if self.master:
                for k, v in loss_scores.items():
                    setattr(self.uid_tracker[uid].train.assigned, k, v)
            self.logger.info(
                f"UID {uid:03d}: Assigned <=> Absolute loss improvement: {loss_scores['absolute']:.6f}"
            )
            self.logger.info(
                f"UID {uid:03d}: Assigned <=> Relative loss improvement: {loss_scores['relative']:.6f}"
            )
            score_status = torch.tensor([1])

        except Exception as e:
            self.logger.info(f"UID {uid:03d}: Error calculating loss score: {e}")

        finally:
            dist.all_reduce(score_status, group=self.gloo_group)
            self.logger.info(f"UID {uid:03d}: Score status {score_status[0].item()}")
            if (score_status[0].item() != self.world_size) and self.master:
                reset_uid_train_scores(self, uid)
                self.uid_tracker[uid].train.updated_time = test_time
            elif self.master:
                # Mark update time for UID
                self.uid_tracker[uid].train.updated_time = test_time

    # Reset model state dict before an evaluation
    set_model_state_dict(
        self.model,
        dict(model_state_dict),
        options=StateDictOptions(
            full_state_dict=False,  # We saved only local shards, so load them back as such
            strict=True,  # Ensure all keys match
        ),
    )

    # Remove stale gradient cache
    cleanup_old_cache(self)


def score_repo(self, uid: int, prefix: str) -> bool:
    """
    Check if the miner's R2 manifest exists and is recent enough.
    """
    try:
        bucket_name = f"{self.config.neuron.global_model_name}-{uid:03d}"
        r2 = get_r2_client(self, uid, donwload_on_all_ranks=False)
        response = r2.head_object(Bucket=bucket_name, Key=f"{prefix}gradients.pt")
        last_modified = (
            email.utils.parsedate_to_datetime(
                response["LastModified"].strftime("%a, %d %b %Y %H:%M:%S GMT")
            )
            if isinstance(response["LastModified"], datetime)
            else email.utils.parsedate_to_datetime(response["LastModified"])
        )

        age_seconds = (datetime.now(timezone.utc) - last_modified).total_seconds()
        return age_seconds < MAX_UPLOAD_INTERVAL
    except Exception as e:
        self.logger.info(f"UID {uid:03d}: Manifest check failed — {e}")
        return False


def benchmark_uids(self):
    """
    Benchmark each UID by checking if their model is valid and up-to-date.
    """
    if (
        self.blocks_since_allreduce < (self.config.neuron.blocks_per_allreduce / 2)
    ) and (self.global_progress.epoch != 0):
        epoch = self.global_progress.epoch - 1
        prefix = f"epoch-{epoch}/"
    else:
        epoch = self.global_progress.epoch
        prefix = ""
    # epoch = self.global_progress.epoch
    prefix = f"epoch-{epoch}/"

    for uid in self.uid_tracker:
        try:
            self.uid_tracker[uid].train.is_valid = score_repo(self, uid, prefix)
        # except (RepositoryNotFoundError, RevisionNotFoundError, OSError) as e:
        #     # self.logger.info(f"UID {uid} benchmarking failed with error {e}. Updating score to 0.")
        #     self.uid_tracker[uid].train.is_valid = False
        except Exception as e:
            self.logger.info(
                f"UID {uid} benchmarking failed with error {e}. Keeping score as is."
            )


def display_rankings(self, uids: list, original_openskill_scores: dict):
    """
    This function prints a table showing this round's UID rankings based off train.assigned.scores and train.random.scores

    Args:
        uids (list): UIDs of miners to be evaluated.
        original_openskill_scores (dict): original OpenSkill ordinal values before update
    """
    # Create a ranking table to display current match rankings
    try:
        # Sort UIDs by current window gradient scores (descending)
        sorted_uids = sorted(
            uids,
            key=lambda uid: (self.uid_tracker[uid].train.score),
            reverse=True,
        )

        try:
            width = os.get_terminal_size().columns
        except Exception:
            width = 0
        os.environ["COLUMNS"] = str(max(200, width))

        rich_table = Table(title=f"Current Round Rankings (Block {self.current_block})")
        rich_table.add_column("Round Rank")
        rich_table.add_column("UID")
        rich_table.add_column("Train Final")
        rich_table.add_column("Train Assigned")
        rich_table.add_column("Train Random")
        rich_table.add_column("Train Random μ")
        rich_table.add_column("Train Random σ")
        rich_table.add_column("Train Random Δ")

        # Add rows to table
        for rank, uid in enumerate(sorted_uids, 1):
            rating = self.openskill_ratings[uid]
            ordinal_before = original_openskill_scores[uid]
            ordinal_after = rating.ordinal()
            ordinal_diff = ordinal_after - ordinal_before

            # Format the diff with color indicators
            diff_str = f"{ordinal_diff:+.4f}"

            rich_table.add_row(
                str(rank),
                str(uid),
                f"{self.uid_tracker[uid].train.score:.6f}",
                f"{self.uid_tracker[uid].train.assigned.score:.6f}",
                f"{self.uid_tracker[uid].train.random.score:.6f}",
                f"{rating.mu:.4f}",
                f"{rating.sigma:.4f}",
                diff_str,
            )

        # Render table to string
        sio = StringIO()
        console = Console(file=sio, width=int(os.environ["COLUMNS"]))
        console.print(rich_table)
        table_str = sio.getvalue()

        self.logger.info(
            f"Current Round Rankings (Block {self.current_block}):\n{table_str}"
        )
    except Exception as e:
        self.logger.info(f"Failed to create Round Rankings Table: {str(e)}")


def update_train_scores(self, uids: list):
    """
    Update selected miners' train.score using the following method:
    1. Calculates train.random.score using the OpenSkill ratings of the relative improvement of a model's loss
    after applying a miner's pseudo gradients on a random unseen dataset.
    2. Calculates train.assigned.score which is a binary indicator of wether relative improvement of a model's
    loss on an unseen dataset is lower than the relative improvement of a model's loss on the miner's assigned dataset.
    3. Calculates train.score as the product of train.random.score and train.assigned.score.

    The OpenSkill rating system (https://arxiv.org/pdf/2401.05451) provides a probabilistic skill rating
    that accounts for uncertainty and relative performance between peers. Ratings are updated using the
    PlackettLuce model where higher gradient scores indicate better performance.

    Args:
        uids (list): UIDs of miners to be evaluated.
    """
    # Store original Openskill ordinal values to calculate diff after update
    original_openskill_scores = {}
    for uid in uids:
        if uid in self.openskill_ratings:
            original_openskill_scores[uid] = float(
                self.openskill_ratings[uid].ordinal()
            )
        else:
            # For new peers without previous ratings
            original_openskill_scores[uid] = 0.0

    # Calculate ranks based on gradient scores (lower rank = better performance)
    scores = [self.uid_tracker[uid].train.random.relative for uid in uids]

    # Create teams list for OpenSkill where each miner is a team of one
    teams = [[self.openskill_ratings[uid]] for uid in uids]

    # Rate each teams using scores (higher score = better performance)
    rated_teams = self.openskill_model.rate(teams, scores=scores)

    # Store updated ratings
    for i, uid in enumerate(uids):
        self.openskill_ratings[uid] = rated_teams[i][0]

        # Log updated OpenSkill values
        self.uid_tracker[uid].train.openskill_rating.mu = float(
            self.openskill_ratings[uid].mu
        )
        self.uid_tracker[uid].train.openskill_rating.sigma = float(
            self.openskill_ratings[uid].sigma
        )
        self.uid_tracker[uid].train.random.score = float(
            self.openskill_ratings[uid].ordinal()
        )
        self.uid_tracker[uid].train.assigned.score = (
            (1 - self.config.neuron.assigned_loss_score_moving_average_alpha)
            * self.uid_tracker[uid].train.assigned.score
        ) + (
            self.config.neuron.assigned_loss_score_moving_average_alpha
            * (
                self.uid_tracker[uid].train.assigned.absolute
                > self.uid_tracker[uid].train.random.absolute
            )
        )
        self.uid_tracker[uid].train.score = max(
            0,
            (
                self.uid_tracker[uid].train.random.score
                * self.uid_tracker[uid].train.assigned.score
            ),
        )

    display_rankings(self, uids, original_openskill_scores)

    self.logger.info(
        f"Updated train scores for {len(uids)} UIDs based on OpenSkill ratings of gradient scores",
    )


def update_all_reduce_scores(self):
    """
    Update all_reduce.score based on the allreduce_status_dict.
    """
    try:
        if self.allreduce_status_dict != {}:
            for uid in self.allreduce_status_dict.keys():
                if (self.allreduce_status_dict[uid] == "SUCCESS") or (
                    self.allreduce_status_dict[uid] == "NON_PARTICIPATING"
                ):
                    score = 1
                else:
                    score = 0
                if int(uid) not in self.uid_tracker:
                    continue
                if self.uid_tracker[int(uid)].all_reduce.score != score:
                    self.uid_tracker[int(uid)].all_reduce.count += 1
                self.uid_tracker[int(uid)].all_reduce.score = score
    except Exception as e:
        self.logger.info(f"Error {e} updating all_reduce scores")


def update_total_scores(self):
    """
    Update total.scores for each UID based on train.score and all_reduce.score.
    """
    # Update AllReduce stats from the latest round
    update_all_reduce_scores(self)

    # Sort uid tracker
    self.uid_tracker = dict(sorted(self.uid_tracker.items()))

    # Normalise each type of reward
    train_scores = [self.uid_tracker[uid].train.score for uid in self.uid_tracker]
    all_reduce_scores = [
        self.uid_tracker[uid].all_reduce.score for uid in self.uid_tracker
    ]
    repo_valid_scores = [
        self.uid_tracker[uid].train.is_valid for uid in self.uid_tracker
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
        train_score = uid_data.train.score
        all_reduce_score = uid_data.all_reduce.score
        repo_valid_score = uid_data.train.is_valid

        normalized_train_score = (
            TRAIN_SCORE_WEIGHT * train_score
        ) / train_scores_normalised
        normalized_all_reduce_score = (
            ALL_REDUCE_SCORE_WEIGHT * all_reduce_score
        ) / all_reduce_scores_normalised
        uid_data.total.score = (
            normalized_train_score + normalized_all_reduce_score
        ) * repo_valid_score

    # Add metrics reporting
    self.report_train_scores()
