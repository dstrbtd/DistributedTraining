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
import threading
import time
import traceback
import torch.distributed as dist

import bittensor as bt

from enum import Enum
from distributed_training.base.neuron import BaseNeuron
from distributed_training.utils.chain import log_r2_to_chain
from distributed_training.utils.misc import get_bandwidth
from distributed_training.utils.state_loader import load_state_from_peer
from distributed_training.utils.progress_tracker import get_progress


class TrainingStatus(Enum):
    ERROR = "❗ | Error"
    RUNNING = "🏋️ | Training"
    STOPPED = "😴 | Stopped"
    PAUSED = "🔄 | Paused"


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            self.logger.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            self.logger.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        if self.master:
            # The axon handles request processing, allowing validators to send this miner requests.
            self.axon = bt.axon(
                wallet=self.wallet,
                config=self.config,
                port=self.config.axon.port,
                ip=self.config.axon.ip,
                external_ip=self.config.axon.external_ip,
                external_port=self.config.axon.external_port,
            )

            # Attach determiners which functions are called when servicing a request.
            self.logger.info("Attaching forward function to miner axon.")
            self.axon.attach(
                forward_fn=self.is_alive,
                blacklist_fn=self.blacklist_is_alive,
                # priority_fn=self.priority,
            ).attach(
                forward_fn=self.all_reduce,
                blacklist_fn=self.blacklist_all_reduce,
            )
            self.logger.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        # self.config.neuron.disable_set_weights = True

        # Log PeerID to chain flag
        self.r2_credentials_logged_to_chain = False

    # def run(rank, self, world_size):
    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """
        self.logger.info("Synced metagraph")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            dist.barrier()
            self.resume_training()

            while not self.should_exit:
                try:
                    if self.master:
                        if self.r2_credentials_logged_to_chain is False:
                            log_r2_to_chain(self)

                        if not self.config.neuron.dont_wandb_log:
                            if self.event != {}:
                                self.event.update(self.get_miner_info())
                                try:
                                    self.bandwidth = get_bandwidth()
                                    self.event.update(self.bandwidth)
                                except Exception:
                                    self.logger.debug("Error getting bandwidth metrics")
                                if self.master:
                                    self.wandb.log(self.event)
                                self.event = {}

                    self.logger.debug(
                        "self.training_active.set()",
                        self.training_active.is_set(),
                        "pre dataset",
                    )
                    # Wait if training is paused
                    self.training_active.wait()

                    self.set_current_block_across_ranks()
                    block_at_start = self.current_block
                    self.logger.debug(
                        f"Block passed to dataloader and block list: {block_at_start}"
                    )

                    self.logger.debug(":pages: Fetching fineweb-edu pages")
                    dataset = self.training_loop.run_until_complete(
                        self.fetch_training_data(block_at_start)
                    )

                    # Wait if training is paused
                    self.logger.debug(
                        "self.training_active.wait()",
                        self.training_active.is_set(),
                        "post dataset",
                    )
                    self.training_active.wait()

                    if self.master:
                        self.model.config.block_list.append(block_at_start)
                    self._process_training_batch(dataset)

                except Exception as e:
                    self.logger.warning(f"Training Loop Failed with error: {e}")
                    self.training_status = TrainingStatus.ERROR
                    self.training_error = str(e)
                    break

            # Await the training task to ensure it completes before exiting
            self.training_status = TrainingStatus.STOPPED

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.should_exit = True
            if self.master:
                self.axon.stop()
            self.logger.success(
                ":white_heavy_check_mark: Miner killed by keyboard interrupt."
            )
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            self.logger.error(traceback.format_exc())

    def load_state(self, reset_last_allreduce_block=False):
        self.global_progress.epoch = get_progress(self, "global")[0]
        if self.local_progress.epoch != self.global_progress.epoch:
            self.logger.info(
                f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
            )
            self.pause_training()
            # If there's an ongoing upload, check if it's done
            while self.current_upload_future and not self.current_upload_future.done():
                self.logger.info(
                    "Previous upload still in progress. Waiting until upload is complete."
                )
                time.sleep(1)
            if self.global_progress.epoch == 0:
                load_state_from_peer(self, epoch=self.global_progress.epoch)
            else:
                load_state_from_peer(
                    self,
                    uid=self.uid,
                    epoch=self.global_progress.epoch,
                )
            self.model.config.block_list = []
            self.resume_training()
        if reset_last_allreduce_block:
            self.last_allreduce_block = None

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            self.logger.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            self.logger.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            self.logger.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            self.logger.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        # self.run_in_background_thread()
        self.run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        self.logger.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
