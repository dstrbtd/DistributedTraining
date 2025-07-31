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

import bittensor as bt

from enum import Enum
from distributed_training.base.neuron import BaseNeuron
from distributed_training.utils.chain import log_peerid_to_chain
from distributed_training.utils.misc import get_bandwidth
from distributed_training.utils.state_loader import load_state_from_peer
from distributed_training.utils.progress_tracker import get_global_epoch


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
        self.peer_id_logged_to_chain = False

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

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        self.logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid} and port: {self.axon.port}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()
        self.logger.info(f"Miner starting at block: {self.block}")

        # Starting training thread
        self.start_continuous_training()

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.metagraph.last_update[self.uid]
                    < self.config.neuron.epoch_length
                ):
                    if self.peer_id_logged_to_chain is False:
                        log_peerid_to_chain(self)

                    if not self.config.neuron.dont_wandb_log:
                        if self.event != {}:
                            self.event.update(self.get_miner_info())
                            try:
                                self.bandwidth = get_bandwidth()
                                self.event.update(self.bandwidth)
                            except Exception:
                                self.logger.debug("Error getting bandwidth metrics")
                            self.wandb.log(self.event)
                            self.event = {}

                    if not self.all_reduce_success_status:
                        wait_time = (
                            self.allreduce_timeout
                            + self.upload_state_duration
                            - time.perf_counter()
                            + self.all_reduce_start_time
                        )
                        self.logger.info(
                            f"Waiting {int(wait_time)} seconds until validator complete the all_reduce"
                        )
                        # Wait for the master validator to upload new global model
                        time.sleep(wait_time)
                        # Check if master validator has failed to all_reduce
                        self.global_progress.epoch = get_global_epoch(self)
                        if self.local_progress.epoch > self.global_progress.epoch:
                            self.logger.info(
                                f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
                            )
                            load_state_from_peer(
                                self,
                                epoch=self.global_progress.epoch,
                            )
                        else:
                            load_state_from_peer(
                                self,
                                repo_id=self.config.neuron.local_model_name,
                                epoch=self.global_progress.epoch,
                            )
                        self.model.config.block_list = []
                        self.resume_training()
                        self.all_reduce_success_status = True
                    else:
                        if (self.last_allreduce_block is not None) and (
                            (time.perf_counter() - self.all_reduce_start_time)
                            > (self.allreduce_timeout + self.upload_state_duration)
                        ):
                            self.load_state(reset_last_allreduce_block=True)
                        elif (self.last_allreduce_block is None) and (
                            self.current_block % self.config.neuron.epoch_length == 0
                        ):
                            self.load_state(reset_last_allreduce_block=False)

                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

            # Await the training task to ensure it completes before exiting

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.should_exit = True
            self.axon.stop()
            self.logger.success(
                ":white_heavy_check_mark: Miner killed by keyboard interrupt."
            )
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            self.logger.error(traceback.format_exc())

    def load_state(self, reset_last_allreduce_block=False):
        self.global_progress.epoch = get_global_epoch(self)
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
                    repo_id=self.config.neuron.local_model_name,
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
