import asyncio
import datetime as dt
import random
import requests
import traceback
from typing import List

import bittensor as bt
import distributed_training
import numpy as np
from bittensor.core.chain_data import decode_account_id
from hivemind.p2p import PeerID
from hivemind.utils.timed_storage import ValueWithExpiration
from huggingface_hub import list_repo_commits
from distributed_training.utils.state_loader import check_model_exists


async def check_uid(self, dendrite, axon, uid, epoch=None):
    try:
        response = await dendrite(
            axon,
            distributed_training.protocol.IsAlive(),
            deserialize=False,
            timeout=2.3,
        )
        if response.is_success:
            if (epoch is not None) and (response.epoch == epoch):
                self.logger.trace(f"UID {uid} is active and on epoch {epoch}")
                return True
            elif (epoch is not None) and (response.epoch != epoch):
                self.logger.trace(f"UID {uid} is active but not on epoch {epoch}")
                return False
            else:
                self.logger.trace(f"UID {uid} is active.")
                return True
        else:
            self.logger.trace(f"UID {uid} is not active.")
            return False
    except Exception as e:
        self.logger.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        # loop.close()
        return False


async def check_uid_availability(
    self,
    dendrite,
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    epoch: int = None,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.

    if not metagraph.axons[uid].is_serving:
        return False

    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False

    # Filter for miners that are processing other responses
    if not await check_uid(self, dendrite, metagraph.axons[uid], uid, epoch):
        return False
    # Available otherwise.
    return True


async def get_random_uids(
    self, dendrite, k: int, exclude: List[int] = None, epoch: int = None
) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []
    uids = [i for i in range(self.metagraph.n)]
    random.shuffle(uids)

    responses = []
    attempt = 0
    limit = self.config.neuron.uid_isalive_limit
    while (sum(responses) < k) and (
        (attempt < (int(self.metagraph.n / limit) - 1)) or (attempt == 0)
    ):
        tasks = []
        if limit > int(self.metagraph.n):
            limit = int(self.metagraph.n)

        for i in range((attempt * limit), (attempt * limit) + limit):
            # The dendrite client queries the network.
            tasks.append(
                check_uid_availability(
                    self,
                    dendrite,
                    self.metagraph,
                    uids[i],
                    self.config.neuron.vpermit_tao_limit,
                    epoch,
                )
            )
        responses += await asyncio.gather(*tasks)
        attempt += 1

    for i, response in enumerate(responses):
        if response == False:
            self.failed_is_alive_counter[uids[i]] += 1
        else:
            self.failed_is_alive_counter[uids[i]] = 0

    for uid, uid_is_available in zip(uids, (responses)):
        uid_is_not_excluded = exclude is None or uid not in exclude
        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        uids = np.array(available_uids)
    else:
        uids = np.array(random.sample(available_uids, k))
    return uids


def get_next_uids_manual(self, k: int = 25) -> List[int]:
    try:
        # Rank miners based off train_similarity_score_last_updated
        self.uid_tracker = dict(
            sorted(
                self.uid_tracker.items(),
                key=lambda item: (
                    not item[1].train.is_valid,
                    item[1].train.updated_time,
                ),
            )
        )
        uids = list(self.uid_tracker.keys())
        uids = uids[:k]
        return uids

    except Exception as e:
        self.logger.info(f"Error getting UID manually: {e}")


def get_next_uid_api(self):
    try:
        response = requests.get(
            url=self.uid_api_url, headers={"Authorization": self.uid_api_get_token}
        )
        uids = response.json()["uids"]

        assert uids != self.miner_uids
        assert type(uids) == list
        assert all(isinstance(uid, int) for uid in uids)
        return uids
    except Exception as e:
        self.logger.info(
            f"Error {e} getting UID from: {self.uid_api_url}. Attempting to get UID manually."
        )
        uids = get_next_uids_manual(self, k=self.config.neuron.sample_size)

    return uids


def post_next_uid_api(self):
    uids = get_next_uids_manual(self, k=self.config.neuron.sample_size)
    try:
        response = requests.post(
            url=self.uid_api_url,
            json={"uids": uids},
            headers={"Authorization": self.uid_api_post_token},
        )
        if response.status_code != 200:
            raise Exception(f"UID post request failed with error: {e}")
    except Exception as e:
        self.logger.info(
            f"Error {e} getting UID from: {self.uid_api_url}. Attempting to get UID manually."
        )


def update_run_peerid_list(self):
    prefix = self.grad_averager.matchmaking_kwargs["prefix"]
    metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or (
        {},
        None,
    )
    self.run_peer_id_list = [
        str(PeerID(peer_id))
        for peer_id, info in metadata.items()
        if isinstance(info, ValueWithExpiration)
        and isinstance(info.value, (float, int))
    ]


def decode_metadata(encoded_ss58: tuple, metadata: dict) -> tuple[str, str]:
    decoded_key = decode_account_id(encoded_ss58[0])
    commitment = metadata["info"]["fields"][0][0]
    bytes_tuple = commitment[next(iter(commitment.keys()))][0]
    return decoded_key, bytes(bytes_tuple).decode()


def map_uid_to_peerid(self):
    result = {}
    try:
        subtensor = bt.subtensor(config=self.config)
        result = subtensor.substrate.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[self.config.netuid],
            block_hash=None,
        )
        hotkey_to_uid = dict(zip(self.metagraph.hotkeys, self.metagraph.uids.tolist()))
    except Exception as e:
        self.logger.info(f"Error {e} when querying UID commitments")

    for key, value in result:
        try:
            hotkey, metadata = decode_metadata(key, value.value)
            if hotkey not in hotkey_to_uid:
                continue

            uid = hotkey_to_uid[hotkey]
            last_updated_block = value.value.get("block", 0)
            if last_updated_block is None:
                last_updated_block = 0
            concatenated = eval(metadata)

            if "peer_id" not in concatenated:
                self.logger.debug(
                    f"Invalid commitment for UID {uid}: peer_id not in commitment metadata"
                )
                continue
            if "model_huggingface_id" not in concatenated:
                self.logger.debug(
                    f"Invalid commitment for UID {uid}: model_huggingface_id not in commitment metadata"
                )
            if concatenated["peer_id"] != self.uid_tracker[uid].all_reduce.peer_id:
                uid_peerid_metadata = [
                    metadata.all_reduce.peer_id
                    for key, metadata in self.uid_tracker.items()
                    if key != uid
                ]
                if concatenated["peer_id"] not in uid_peerid_metadata:
                    self.uid_tracker[uid].all_reduce.peer_id = concatenated["peer_id"]
                    self.uid_tracker[
                        uid
                    ].chaindata.last_updated_block = last_updated_block
                else:
                    uid_list = [
                        uid
                        for uid, metadata in self.uid_tracker.items()
                        if metadata.all_reduce.peer_id == concatenated["peer_id"]
                    ]
                    for uid_i in uid_list:
                        if (
                            self.uid_tracker[uid_i].chaindata.last_updated_block
                            is not None
                        ) and (
                            self.uid_tracker[uid_i].chaindata.last_updated_block
                            > last_updated_block
                        ):
                            self.uid_tracker[uid_i].chaindata.last_updated_block = 0
                            self.uid_tracker[uid_i].train.model_id = None
                            self.uid_tracker[uid_i].all_reduce.peer_id = None
                        else:
                            self.uid_tracker[uid].chaindata.last_updated_block = 0
                            self.uid_tracker[uid].train.model_id = None
                            self.uid_tracker[uid].all_reduce.peer_id = None
            if (
                concatenated["model_huggingface_id"]
                != self.uid_tracker[uid].train.model_id
            ):
                self.uid_tracker[uid].train.model_id = concatenated[
                    "model_huggingface_id"
                ]
                uid_peerid_metadata = [
                    metadata.train.model_id
                    for key, metadata in self.uid_tracker.items()
                    if key != uid
                ]
                if concatenated["model_huggingface_id"] not in uid_peerid_metadata:
                    self.uid_tracker[uid].train.model_id = concatenated[
                        "model_huggingface_id"
                    ]
                    self.uid_tracker[
                        uid
                    ].chaindata.last_updated_block = last_updated_block
                else:
                    uid_list = [
                        uid
                        for uid, metadata in self.uid_tracker.items()
                        if metadata.train.model_id
                        == concatenated["model_huggingface_id"]
                    ]
                    for uid_i in uid_list:
                        if (
                            self.uid_tracker[uid_i].chaindata.last_updated_block
                            is not None
                        ) and (
                            self.uid_tracker[uid_i].chaindata.last_updated_block
                            > last_updated_block
                        ):
                            self.uid_tracker[uid_i].chaindata.last_updated_block = 0
                            self.uid_tracker[uid_i].train.model_id = None
                            self.uid_tracker[uid_i].all_reduce.peer_id = None
                        else:
                            self.uid_tracker[uid].chaindata.last_updated_block = 0
                            self.uid_tracker[uid].train.model_id = None
                            self.uid_tracker[uid].all_reduce.peer_id = None

            self.logger.debug(f"Retrieved commitment for UID {uid}: {metadata}")

        except Exception as e:
            self.logger.debug(f"Failed to decode commitment for UID {uid}: {e}")
            continue

    self.logger.debug("Finished extracting commitments for all uids")
