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
import torch
import torch.distributed as dist
import bittensor as bt
import numpy as np

from distributed_training import __run__
from distributed_training.averaging.exceptions import GradientAveragingError
from distributed_training.utils.misc import get_bandwidth
from distributed_training.utils.progress_tracker import (
    get_progress,
)
from distributed_training.utils.state_loader import (
    load_state_from_peer,
    upload_new_state,
)
from distributed_training.utils.uids import (
    get_next_uid_api,
    post_next_uid_api,
    get_random_uids,
    map_uid_to_peerid,
)
from distributed_training.validator.reward import (
    benchmark_uids,
    score_uids,
    update_total_scores,
    update_train_scores,
    evaluate_model,
)
from distributed_training.averaging.averagers import (
    compute_and_load_pseudo_grad_into_averager,
    apply_optimizer_parameters,
)
from distributed_training.utils.dist import gloabl_dist_checkpoint
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
)


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    if self.master:
        # Evaluate wether to run an AllReduce or validate HF miner states
        if self.step % 2 == 0:
            map_uid_to_peerid(self)

        # Benchmark UIDs
        benchmark_uids(self)

    # Get number of blocks since last allreduce
    self.set_current_block_across_ranks()
    self.blocks_since_allreduce = self.current_block - self.last_allreduce_block
    self.should_all_reduce = (
        self.blocks_since_allreduce >= self.config.neuron.blocks_per_allreduce
    )
    self.logger.info(
        f"Current block {self.current_block} | Blocks Since Last AllReduce: {self.blocks_since_allreduce} | Should AllReduce: {self.should_all_reduce}"
    )
    if (
        self.blocks_since_allreduce < (self.config.neuron.blocks_per_allreduce / 2)
    ) and (self.global_progress.epoch != 0):
        epoch = self.global_progress.epoch - 1
    else:
        epoch = self.global_progress.epoch
    # epoch = self.global_progress.epoch

    responses = [[]]

    if self.should_all_reduce:
        if self.master:
            self.event.update({"synapse_type": "all_reduce"})

            self.peerids_to_uids = {
                str(value.all_reduce.peer_id): key
                for key, value in self.uid_tracker.items()
            }
        if self.uid == self.master_uid:
            if self.master:
                # Master validator coordinates AllReduce and queries miners
                sample_size = int(self.metagraph.n)

                min_sample_size = self.config.neuron.min_group_size * 2
                self.miner_uids = []
                # Get active miners
                while len(self.miner_uids) < min_sample_size:
                    self.logger.info(
                        f"Found {len(self.miner_uids)} UIDs. Attempting to find {min_sample_size - len(self.miner_uids)} more UIDs."
                    )
                    self.miner_uids += (
                        await get_random_uids(
                            self,
                            dendrite=self.dendrite,
                            k=sample_size,
                            epoch=self.local_progress.epoch,
                        )
                    ).tolist()
                    self.miner_uids = list(set(self.miner_uids))
                    if (len(self.miner_uids) < min_sample_size) and (
                        min_sample_size > 3
                    ):
                        min_sample_size = min_sample_size - 1
            dist.barrier()
        else:
            # For non-master validators
            self.logger.info(
                f"Waiting {self.allreduce_timeout + self.upload_state_duration} seconds whilst master UID completes all reduce."
            )
            time.sleep(self.allreduce_timeout + self.upload_state_duration)
            self.miner_uids = []
            self.set_current_block_across_ranks()
            # Reset allreduce block tracker
            self.last_allreduce_block = self.current_block
            return responses

        if self.master:
            for uid in self.miner_uids:
                self.uid_tracker[
                    uid
                ].train.revision = f"{__run__}.{self.global_progress.epoch}.{get_progress(self, 'local', uid=uid, donwload_on_all_ranks=False)[1]}"
            self.miner_uids = np.flip(
                np.array(
                    [
                        self.miner_uids[i]
                        for i in np.argsort(
                            [
                                # self.metagraph.incentive[i].item()
                                # self.uid_tracker[i].total.score
                                # * self.uid_tracker[i].train.is_valid
                                int(self.uid_tracker[i].train.revision.split(".")[-1])
                                for i in self.miner_uids
                            ]
                        )
                    ]
                )
            )
            alive_uids = self.miner_uids
            miner_uids = torch.tensor(
                self.miner_uids.tolist()[: self.config.neuron.min_group_size * 2]
            )
            self.event.update({"UIDs": self.miner_uids})
            self.logger.info(f"UIDs:  {self.miner_uids}")
        else:
            miner_uids = torch.tensor([0] * self.config.neuron.sample_size)

        dist.broadcast(miner_uids, src=0, group=self.gloo_group)
        self.miner_uids = miner_uids.tolist()

        try:
            #     top_uid_index = 0
            #     while True:
            #         if top_uid_index < len(self.miner_uids):
            #             top_uid = self.miner_uids[top_uid_index]
            #         else:
            #             top_uid = 212
            #         self.local_progress.epoch = self.global_progress.epoch

            #         self.logger.info(f"Top UID identified as: {top_uid}")

            #         self.local_progress.inner_step = get_progress(
            #             self, local_or_global="local", uid=top_uid
            #         )[1]
            #         top_uid_revision = f"{__run__}.{self.local_progress.epoch}.{self.local_progress.inner_step}"
            #         load_state_from_peer(
            #             self,
            #             uid=top_uid,
            #             revision=top_uid_revision,
            #         )
            #         if self.scheduler.__dict__["_step_count"] != 0:
            #             break
            #         else:
            #             top_uid_index += 1

            all_reduce_success_status = True

            # Create and normalize gradients
            self.logger.info(":wait: Starting Compute Pseudo Gradients")
            compute_and_load_pseudo_grad_into_averager(self)
            self.logger.info(":wait: Finished Compute Pseudo Gradients")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            if self.master:
                (
                    all_reduce_success_status,
                    results,
                    initial_weights,
                ) = await self.avg_handler.run_validator_allreduce(
                    timeout=self.allreduce_timeout,
                    wallet=self.wallet,
                    metagraph=self.metagraph,
                    peerids_to_uids=self.peerids_to_uids,
                    miner_uids=self.miner_uids,
                    master=self.uid == self.master_uid,
                    block=self.current_block,
                    min_group_size=self.config.neuron.min_group_size,
                )

            if not gloabl_dist_checkpoint(all_reduce_success_status, self.gloo_group):
                raise Exception(f"All Reduce Failed At Checkpoint 1")

            # Normalize averaged gradients
            try:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            except Exception:
                all_reduce_success_status = False

            if not gloabl_dist_checkpoint(all_reduce_success_status, self.gloo_group):
                raise Exception(f"All Reduce Failed At Checkpoint 2")

            if self.master:
                # Perform offloaded outer optimization steps
                self.outer_optimizer.step()
                self.logger.info(
                    ":white_heavy_check_mark: Finished Outer Optimizer Step."
                )

            try:
                (
                    total_loss_before,
                    _,
                    n_batches_sampled_before,
                ) = await evaluate_model(
                    self,
                    model=self.model,
                    blocks=[self.current_block],
                    uid=self.uid,
                    samples=None,
                    n_pages=2,
                )
                average_loss_before = total_loss_before / n_batches_sampled_before

                self.logger.info(
                    ":white_heavy_check_mark: Finished Outer Optimizer Step."
                )
            except Exception:
                all_reduce_success_status = False

            if not gloabl_dist_checkpoint(all_reduce_success_status, self.gloo_group):
                raise Exception(f"All Reduce Failed At Checkpoint 3")

            apply_optimizer_parameters(self)

            if self.master:
                self.logger.info("Validate Weights After Opt Step")
                # Validate weight updates
                await self.avg_handler._validate_weight_update(
                    initial_weights, self.current_block
                )

            try:
                self.logger.info(f"Apply optimizer parameters to model")
                (
                    total_loss_after,
                    _,
                    n_batches_sampled_after,
                ) = await evaluate_model(
                    self,
                    model=self.model,
                    blocks=[self.current_block],
                    uid=self.uid,
                    samples=None,
                    n_pages=2,
                )
                average_loss_after = total_loss_after / n_batches_sampled_after

                if (
                    (average_loss_after - average_loss_before) / (average_loss_after)
                ) > 0.25:
                    all_reduce_success_status = False
                    raise Exception(
                        f"Average Loss After All Reduce {average_loss_after} > 25% higher than Average Loss Before {average_loss_before}"
                    )
            except Exception:
                all_reduce_success_status = False

            if not gloabl_dist_checkpoint(all_reduce_success_status, self.gloo_group):
                raise Exception(f"All Reduce Failed At Checkpoint 4")

            self.set_current_block_across_ranks()
            # Reset allreduce block tracker
            self.last_allreduce_block = self.current_block
            # Update state after successful allreduce
            self.local_progress.epoch += 1
            self.local_progress.samples_accumulated = 0

            if self.master:
                # Update scoring based on allreduce participation
                (
                    self.allreduce_scores,
                    self.allreduce_status_dict,
                    self.event,
                    successful_peers_count,
                ) = self.avg_handler.calculate_allreduce_scores(
                    participating_peers=results["participating_peers"],
                    failed_peers=results["failed_peers"],
                    alive_uids=alive_uids,
                    modes=results["modes"],
                    bandwidths=results["bandwidths"],
                    peerids_to_uids=self.peerids_to_uids,
                    event=self.event,
                    metagraph=self.metagraph,
                )

            model_state_options = StateDictOptions(
                full_state_dict=True,  # gather a full (HF-style) state dict
                cpu_offload=True,  # offload to host RAM (no GPU OOM)
            )
            model_state = get_model_state_dict(self.model, options=model_state_options)

            inner_optimizer_options = StateDictOptions(
                full_state_dict=False, cpu_offload=True
            )
            inner_optimizer_state = get_optimizer_state_dict(
                self.model, self.inner_optimizer, options=inner_optimizer_options
            )
            self.logger.info(f"Extracted Optimizer & Model State Dict")

            # Upload new global state to HF
            upload_new_state(
                self,
                self.local_progress.epoch,
                results if self.master else {},
                model_state,
                inner_optimizer_state,
                self.inner_optimizer.param_groups[0]["lr"],
                self.current_block,
            )

            if self.master:
                update_total_scores(self)

                try:
                    # ---- Report allreduce metrics to dashboard ---
                    participating_count = len(results["participating_peers"])
                    success_rate = 0.0
                    if participating_count > 0:
                        success_rate = successful_peers_count / participating_count

                    avg_bandwidth = None
                    if results["bandwidths"]:
                        valid_bandwidths = [
                            b for b in results["bandwidths"] if b is not None
                        ]
                        if valid_bandwidths:
                            avg_bandwidth = sum(valid_bandwidths) / len(
                                valid_bandwidths
                            )

                    self.report_allreduce_scores(
                        op_id=self.current_block,
                        epoch=self.local_progress.epoch,
                        validator_uid=self.uid,
                        success_rate=success_rate,
                        duration=results["duration"],
                        participating_miners_count=len(results["participating_peers"]),
                        failed_miners_count=len(results["failed_peers"]),
                        bandwidth=avg_bandwidth,
                    )
                    # -------

                except Exception as e:
                    self.logger.info(
                        f"Error reporting allreduce metrics to dashboard {e}"
                    )

                self.all_reduce_success_status = (
                    True if all_reduce_success_status_tensor[0].item() == 1 else False
                )
            else:
                raise GradientAveragingError("Unsuccessful AllReduce Step")

        except Exception as e:
            self.all_reduce_success_status = False
            self.logger.error(f"All Reduce failed with error {e}")
            return

        finally:
            upload_update_completion_tensor = (
                torch.tensor([1])
                if self.all_reduce_success_status is True
                else torch.tensor([0])
            )
            dist.broadcast(
                upload_update_completion_tensor, src=0, group=self.gloo_group
            )
            if upload_update_completion_tensor[0].item() != 1:
                self.logger.info("Failed to completed allreduce & upload process")
                self.global_progress.epoch = get_progress(self, "global")[0]
                self.all_reduce_success_status = False
                self.last_allreduce_block += int(
                    self.config.neuron.blocks_per_allreduce / 10
                )
                return
            else:
                self.logger.info("Succesfully completed allreduce & upload process")
                self.all_reduce_success_status = True
                self.config.neuron.blocks_per_allreduce = 750

    else:
        if self.master:
            # If running HF validation round, only call one UID each step
            self.event.update({"synapse_type": "train"})
            miner_uids = torch.tensor(
                get_next_uid_api(
                    self,
                    epoch,
                )
            )
        else:
            miner_uids = torch.tensor([0] * self.config.neuron.sample_size)

        dist.broadcast(miner_uids, src=0, group=self.gloo_group)
        self.miner_uids = miner_uids.tolist()

        # Early return if no active miners found
        if len(self.miner_uids) == 0:
            self.logger.info("No Active Miners Found This Step.")
            return responses

        if self.master:
            self.event.update({"UIDs": self.miner_uids})
        self.logger.info(f"UIDs:  {self.miner_uids}")

        await score_uids(self, epoch, self.miner_uids)

        if self.master:
            # Update train.scores for each UID
            update_train_scores(self, self.miner_uids)

            # Update total.scores for each UID
            update_total_scores(self)

            # Post
            if self.uid == self.master_uid:
                post_next_uid_api(self, epoch)

    if self.master:
        self.event.update(
            {
                "uids": self.miner_uids,
                "learning_rate": self.learning_rate,
                "average_miner_loss": self.average_loss,
                "local_epoch": self.local_progress.epoch,
                "global_epoch": self.global_progress.epoch,
                "local_samples_accumulated": self.local_progress.samples_accumulated,
                "global_samples_accumulated": self.global_progress.samples_accumulated,
            }
        )
        self.event.update(
            {
                "uid_" + str(key): self.uid_tracker[key].model_dump()
                for key in self.uid_tracker.keys()
            }
        )

        # Update scores
        self.update_scores(self.miner_uids)

        self.event.update(self.get_validator_info())

        try:
            self.event.update(get_bandwidth())
        except Exception:
            self.logger.info("Error getting bandwidth metrics")

    return responses
