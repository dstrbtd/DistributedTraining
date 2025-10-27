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
import threading
import os
from traceback import print_exception
from typing import List
import torch

import bittensor as bt
import numpy as np

from distributed_training.base.neuron import BaseNeuron
from distributed_training.utils.chain import log_r2_to_chain
from distributed_training.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from distributed_training.utils.progress_tracker import UidTracker, get_progress
from distributed_training.utils.state_loader import load_state_from_peer
from distributed_training.validator.reward import update_total_scores
from openskill.models import PlackettLuce
from torch import distributed as dist


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        if self.master:
            # Save a copy of the hotkeys to local memory.
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

            # Dendrite lets us send messages to other nodes (axons) in the network.
            self.dendrite = bt.dendrite(wallet=self.wallet)
            self.logger.info(f"Dendrite: {self.dendrite}")

            # Set up initial scoring weights for validation
            self.logger.info("Building validation weights.")
            self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

            # Initialize openskill_model before loading state
            self.openskill_model = PlackettLuce(
                beta=self.config.neuron.openskill_beta,
                tau=self.config.neuron.openskill_tau,
            )
            self.openskill_ratings = {
                int(uid): self.openskill_model.rating(name=str(uid))
                for uid in range(self.metagraph.n)
            }

            # Initialize uid_tracker
            self.uid_tracker = {
                uid: UidTracker(uid=uid) for uid in self.metagraph.uids.tolist()
            }

            # Load current state
            self.logger.debug("load_state()")
            self.load_state()

            # Set event dictionary
            self.event = {}

            # Init sync with the network. Updates the metagraph.
            self.sync()

            # Serve axon to enable external connections.
            if not self.config.neuron.axon_off:
                self.serve_axon()
            else:
                self.logger.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        # Log PeerID to chain flag
        self.r2_credentials_logged_to_chain = False

    def serve_axon(self):
        """Serve axon to enable external connections."""

        self.logger.info("serving ip to chain...")
        try:
            self.axon = bt.axon(
                wallet=self.wallet,
                config=self.config,
                port=self.config.axon.port,
                ip=self.config.axon.ip,
                external_ip=self.config.axon.external_ip,
                external_port=self.config.axon.external_port,
            )
            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                self.logger.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            self.logger.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        responses = await asyncio.gather(*coroutines)
        return responses

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """
        # Check that validator is registered on the network.
        self.sync()

        if self.master:
            self.logger.info(
                f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
            )

            self.logger.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                if self.master:
                    self.logger.info(f"step({self.step}) block({self.block})")

                    # Init Wandb Event For Step
                    if self.event != {}:
                        self.event = {}

                current_global_epoch = self.global_progress.epoch
                self.global_progress.epoch = get_progress(self, "local")[0]
                if (
                    self.blocks_since_allreduce
                    > (self.config.neuron.blocks_per_allreduce / 2)
                ) and (
                    (self.local_progress.epoch != self.global_progress.epoch)
                    or (not self.all_reduce_success_status)
                ):
                    self.reload_state_event.set()
                    if self.local_progress.epoch != self.global_progress.epoch:
                        self.logger.info(
                            f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
                        )
                    if not self.all_reduce_success_status:
                        self.logger.info(
                            "All Reduce Failed. Loading Latest Model State."
                        )
                    load_state_from_peer(self, epoch=self.global_progress.epoch)
                    # Reset all_reduce success status
                    if not self.all_reduce_success_status:
                        self.set_current_block_across_ranks()
                        self.all_reduce_success_status = True
                        self.last_allreduce_block = self.current_block
                    # Load all_reduce scores if non_master_uid
                    if (
                        self.master
                        and (self.uid != self.master_uid)
                        and (self.global_progress.epoch != current_global_epoch)
                        and (self.should_all_reduce)
                    ):
                        update_total_scores(self)

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                if self.master:
                    # Log to wandb
                    if (
                        not self.config.neuron.dont_wandb_log
                        and "uids" in self.event
                        and len(self.event["uids"]) > 0
                    ):
                        self.wandb.log(self.event)

                    self.step += 1
                    if self.r2_credentials_logged_to_chain is False:
                        log_r2_to_chain(self)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            # self.dht.shutdown()
            # _thread.interrupt_main()
            self.axon.stop()
            self.logger.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            self.logger.error("Error during validation", str(err))
            self.logger.debug(print_exception(type(err), err, err.__traceback__))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            self.logger.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            self.logger.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            self.logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            self.logger.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            self.logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            self.logger.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        self.logger.info(self.scores)
        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            self.logger.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        self.logger.info(raw_weights)
        self.logger.info(f"raw_weights: {raw_weights}")
        self.logger.info(f"raw_weight_uids: {self.metagraph.uids.tolist()}")

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        self.logger.info(f"processed_weights: {processed_weights}")
        self.logger.info(f"processed_weight_uids: {processed_weight_uids}")

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        self.logger.info(f"uint_weights: {uint_weights}")
        self.logger.info(f"uint_uids: {uint_uids}")

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            self.logger.info("set_weights on chain successfully!")
        else:
            self.logger.error("set_weights failed", msg)

        # Log weigths to wandb
        self.event.update(
            {
                f"uid_{processed_weight_uids}.weights": processed_weights
                for processed_weight_uids, processed_weights in zip(
                    processed_weight_uids, processed_weights
                )
            }
        )
        self.logger.info(f"Set weights: {processed_weights}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        self.logger.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        self.logger.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced
                self.uid_tracker[uid] = UidTracker(
                    uid=uid
                )  # reset uid_tracker for this uid
                self.failed_is_alive_counter[uid] = 0

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average
            for uid in range(min_len, len(self.metagraph.hotkeys)):
                self.uid_tracker[uid] = UidTracker(
                    uid=uid
                )  # reset uid_tracker for this uid
                self.failed_is_alive_counter[uid] = 0

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        #  Make sure uid_tracker is sorted by uids
        self.uid_tracker = dict(sorted(self.uid_tracker.items()))
        uids = list(self.uid_tracker.keys())
        rewards = np.array(
            [self.uid_tracker[i].total.score for i in self.uid_tracker.keys()]
        )

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            self.logger.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            self.logger.info(f"rewards: {rewards}, uids_array: {uids_array}")
            self.logger.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = self.scores.copy()
        scattered_rewards[uids_array] = rewards
        for uid in range(len(scattered_rewards)):
            if (
                self.failed_is_alive_counter[uid]
                > self.failed_is_alive_counter_threshold
            ):
                self.logger.info(
                    f"UID {uid} above is_alive_failed_counter threshold. Setting score to 0."
                )
                scattered_rewards[uid] = 0

        self.logger.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = alpha * scattered_rewards + (1 - alpha) * self.scores
        self.logger.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        self.logger.info("Saving validator state.")
        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            failed_is_alive_counter=self.failed_is_alive_counter,
            uid_tracker={
                key: self.uid_tracker[key].model_dump()
                for key in self.uid_tracker.keys()
            },
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        self.logger.info("Loading validator state.")

        if os.path.isfile(self.config.neuron.full_path + "/state.npz"):
            self.logger.info(
                "Pre-saved validator state found in .npz format. Loading validator state."
            )
            # Load the state of the validator from file.
            state = np.load(
                self.config.neuron.full_path + "/state.npz", allow_pickle=True
            )

            self.step = state["step"]
            self.scores[0 : len(state["scores"])] = state["scores"]
            self.hotkeys[0 : len(state["hotkeys"])] = state["hotkeys"]
            if "failed_is_alive_counter" in state:
                self.failed_is_alive_counter = state[
                    "failed_is_alive_counter"
                ].flatten()[0]
            if "uid_tracker" in state:
                uid_tracker_state = state["uid_tracker"].flatten()[0]
                for uid in uid_tracker_state:
                    try:
                        self.uid_tracker[uid] = UidTracker(**uid_tracker_state[uid])
                    except Exception as e:
                        self.logger.info(
                            f"Failed to load saved uid_tracker for UID: {uid} with error: {e}"
                        )
                        self.uid_tracker[uid] = UidTracker(uid=uid)
                    self.openskill_ratings[uid] = self.openskill_model.rating(
                        mu=self.uid_tracker[uid].train.openskill_rating.mu,
                        sigma=self.uid_tracker[uid].train.openskill_rating.sigma,
                        name=str(uid),
                    )

        else:
            self.logger.info("Pre-saved validator state not found.")
