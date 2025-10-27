# With many thanks to Prime-Intellect for inspiration for this code

import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import bittensor as bt
import hivemind
import torch
from transformers import AutoModelForCausalLM
from hivemind.averaging.allreduce import (
    AllreduceException,
    AllReduceRunner,
    AveragingMode,
)
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import CompressionInfo, deserialize_torch_tensor
from hivemind.dht import DHT
from hivemind.dht.dht import DHT
from hivemind.optim.state_averager import TorchOptimizer, TrainingStateAverager
from hivemind.p2p import P2PContext, P2PDaemonError, P2PHandlerError, PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger
from hivemind.utils.asyncio import (
    aiter_with_timeout,
    amap_in_executor,
    as_aiter,
    attach_event_on_finished,
    azip,
    enter_asynchronously,
)
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.timed_storage import DHTExpiration, get_dht_time

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
    set_model_state_dict,
)
import torch.distributed as dist
import gc


GatheredData = Any

hivemind_logger = get_logger(__name__)


class DTAllReduceRunner(AllReduceRunner):
    def __init__(self, peerids_to_uids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.peerids_to_uids = peerids_to_uids
        bt.logging.info(f"PeerID to UID mapping: {self.peerids_to_uids}")

    async def _communicate_with_peer(self, peer_id: PeerID):
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        uid = self.peerids_to_uids.get(str(peer_id), "'''")
        peer_id_abreviated = str(peer_id)[-3:]

        bt.logging.info(
            f"UID:{uid} - PeerID:{peer_id_abreviated} - communicate_with_peer started",
        )
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
            sender_index = self.sender_peer_ids.index(peer_id)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(
                    sender_index, part_index, tensor_part, weight=self.weight
                )
                self.tensor_part_container.register_processed_part(
                    peer_index,
                    part_index,
                    averaged_part - tensor_part,
                )

        else:
            try:
                done_sending = asyncio.Event()
                inputs_aiter = attach_event_on_finished(
                    self._generate_input_for_peer(peer_index, uid, peer_id),
                    done_sending,
                )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - generate_input_for_peer started"
                )
                stream = await self._get_peer_stub(peer_id).rpc_aggregate_part(
                    inputs_aiter
                )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - get_peer_stub finished"
                )

                if self.should_delay_results(self.peer_id):
                    await done_sending.wait()

                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - sending tensors finished"
                )
                part_index = 0

                def _try_deserialize(msg):
                    if msg.code != averaging_pb2.AVERAGED_PART:
                        raise AllreduceException(
                            f"{peer_id_abreviated} sent {averaging_pb2.MessageCode.Name(msg.code)}"
                        )
                    return deserialize_torch_tensor(msg.tensor_part), msg

                async for delta, msg in amap_in_executor(
                    _try_deserialize,
                    aiter_with_timeout(stream, self.reducer_timeout),
                    max_prefetch=self.tensor_part_container.prefetch,
                ):
                    self.tensor_part_container.register_processed_part(
                        peer_index,
                        part_index,
                        delta,
                    )
                    part_index += 1
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - register_processed_part finished"
                )
                if (
                    part_index
                    != self.tensor_part_container.num_parts_by_peer[peer_index]
                ):
                    bt.logging.info(
                        "part_index != self.tensor_part_container.num_parts_by_peer[peer_index]"
                    )
                    raise AllreduceException(
                        f"peer {peer_id_abreviated} sent {part_index} parts, but we expected "
                        f"{self.tensor_part_container.num_parts_by_peer[peer_index]}"
                    )
            except BaseException as e:
                if isinstance(e, Exception):
                    hivemind_logger.debug(
                        f"Caught {repr(e)} when communicating to {peer_id_abreviated}",
                        exc_info=True,
                    )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - Failed to communicate with peers due to error - {e}"
                )
                self.tensor_part_container.register_failed_reducer(peer_index)
                await self._ban_sender(peer_id)
                raise

    def __aiter__(self):
        return self.run()

    async def run(self) -> AsyncIterator[torch.Tensor]:
        """Run all-reduce, return differences between averaged and original tensors as they are computed"""
        pending_tasks = set()
        bt.logging.info("Running AllReducerRunner")
        bt.logging.info(
            f"self.tensor_part_container.num_parts_by_peer {self.tensor_part_container.num_parts_by_peer}"
        )
        if (
            self.tensor_part_container.num_parts_by_peer[
                self.ordered_peer_ids.index(self.peer_id)
            ]
            != 0
        ):
            pending_tasks.add(asyncio.create_task(self._handle_missing_senders()))

        try:
            if len(self.sender_peer_ids) == 0:
                hivemind_logger.debug(
                    f"{self} - finished all-reduce early: all peers are auxiliaries ({self.modes})"
                )
                self.finalize()

            elif self.peer_id in self.sender_peer_ids:
                uid = self.peerids_to_uids.get(str(self.peer_id), "'''")
                peer_id_abreviated = str(self.peer_id)[-3:]

                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} peer_id in sender_peer_ids"
                )

                for peer_id, parts in zip(
                    self.ordered_peer_ids, self.tensor_part_container.num_parts_by_peer
                ):
                    if parts != 0:
                        pending_tasks.add(
                            asyncio.create_task(self._communicate_with_peer(peer_id))
                        )

                bt.logging.info("Succesfully Communicated With All Peers")

                async for (
                    averaged_tensor_delta
                ) in self.tensor_part_container.iterate_output_tensors():
                    yield averaged_tensor_delta  # delta = averaged_tensor - original_tensor

                bt.logging.info("Iterate Output Tensors Finished")

                self.finalize()

                bt.logging.info("Finalize Finished")

            else:  # auxiliary peer
                await self.tensor_part_reducer.finished.wait()
                self.finalize()

        except BaseException as e:
            bt.logging.info(f"All Reduce Runner failed with error {e}")

            self.finalize(exception=e)
            for task in pending_tasks:
                task.cancel()
            raise

        finally:
            for task in pending_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as inner_exc:
                    hivemind_logger.debug(
                        f"Task {task} failed with {inner_exc}", exc_info=True
                    )

    async def _generate_input_for_peer(
        self, peer_index: int, uid: str, peer_id: PeerID
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        peer_id_abreviated = str(self.peer_id)[-3:]
        try:
            parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
            first_part = await anext(parts_aiter)
            yield averaging_pb2.AveragingData(
                code=averaging_pb2.PART_FOR_AVERAGING,
                group_id=self.group_id,
                tensor_part=first_part,
                weight=self.weight,
            )
            bt.logging.info(
                f"UID:{uid} - PeerID:{peer_id_abreviated} - generate_input_for_peer finished"
            )

            async for part in parts_aiter:
                yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)

        except Exception as e:
            hivemind_logger.error(
                f"Error preparing input for peer {self.ordered_peer_ids[peer_index]}: {e}"
            )

            raise e

    async def _ban_sender(self, peer_id: PeerID):
        uid = self.peerids_to_uids.get(str(peer_id), "'''")
        peer_id_abreviated = str(self.peer_id)[-3:]
        bt.logging.info(f"UID:{uid} - PeerID:{peer_id_abreviated} - Banning Peer")

        async with self.banlock:
            if peer_id not in self.banned_senders:
                self.banned_senders.add(peer_id)
                self.tensor_part_reducer.on_sender_failed(
                    self.sender_peer_ids.index(peer_id)
                )


class DTAverager(hivemind.DecentralizedAverager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(
        self,
        gather: Optional[GatheredData] = None,
        scheduled_time: Optional[DHTExpiration] = None,
        weight: Optional[float] = None,
        timeout: Optional[float] = None,
        allow_retries: bool = True,
        require_trigger: bool = False,
        wait: bool = True,
        **kwargs,
    ) -> Union[Optional[Dict[PeerID, GatheredData]], StepControl]:
        if self.mode == AveragingMode.AUX and weight is not None:
            hivemind_logger.warning(
                "Averager is running in auxiliary mode, weight is unused"
            )
        if scheduled_time is None:
            scheduled_time = (
                get_dht_time() + self.matchmaking_kwargs["min_matchmaking_time"]
            )
        if weight is None:
            weight = float(self.mode != AveragingMode.AUX)
        deadline = get_dht_time() + timeout if timeout is not None else float("inf")
        assert (
            isinstance(weight, (int, float)) and weight >= 0
        ), f"Expected a positive int/float, got {type(weight)}"
        assert not (
            wait and require_trigger
        ), "Non-asynchronous step cannot wait for trigger (use wait=False)"
        assert (
            scheduled_time < deadline
        ), "Scheduled start time does not fit within timeout"

        user_data_for_gather = self.serializer.dumps(
            gather
        )  # serialize here to avoid imports in the averager process
        data_for_gather = self.serializer.dumps(
            [self.bandwidth, self.mode.value, user_data_for_gather]
        )
        step = StepControl(
            scheduled_time=scheduled_time,
            deadline=deadline,
            allow_retries=allow_retries,
            weight=weight,
            data_for_gather=data_for_gather,
        )

        future_for_init = MPFuture()

        self._pending_groups_registered = asyncio.Event()
        self._pending_groups_registered.set()

        peerids_to_uids = kwargs.get("peerids_to_uids", {})

        # Default behavior: initiate matchmaking and proceed as originally designed
        self._outer_pipe.send(
            (
                "_step",
                [],
                dict(
                    step=step,
                    future_for_init=future_for_init,
                    peerids_to_uids=peerids_to_uids,
                ),
            )
        )
        step.attach(*future_for_init.result())

        if not require_trigger:
            step.allow_allreduce()
        return step.result() if wait else step

    async def _step(
        self,
        *,
        step: StepControl,
        future_for_init: MPFuture,
        peerids_to_uids: Dict = {},
    ):
        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))

            async def find_peers_or_notify_cancel():
                group_info = await self._matchmaking.look_for_group(step)
                if not step.triggered:
                    step.stage = AveragingStage.AWAITING_TRIGGER
                    await step.wait_for_trigger()
                return group_info

            while not step.done():
                try:
                    self._pending_groups_registered.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP
                    matchmaking_task = asyncio.create_task(
                        find_peers_or_notify_cancel()
                    )
                    check_cancel_task = asyncio.create_task(step.wait_for_cancel())

                    await asyncio.wait(
                        {matchmaking_task, check_cancel_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if step.cancelled():
                        matchmaking_task.cancel()
                        raise asyncio.CancelledError()
                    else:
                        check_cancel_task.cancel()

                    group_info = await matchmaking_task

                    if group_info is None:
                        raise AllreduceException(
                            "Averaging step failed: could not find a group"
                        )

                    with self._register_allreduce_group(group_info):
                        step.stage = AveragingStage.RUNNING_ALLREDUCE
                        step.set_result(
                            await asyncio.wait_for(
                                self._aggregate_with_group(
                                    group_info,
                                    tensor_infos=self.tensor_infos,
                                    weight=step.weight,
                                    peerids_to_uids=peerids_to_uids,
                                    **self.allreduce_kwargs,
                                ),
                                timeout=self._allreduce_timeout,
                            )
                        )
                        # averaging is finished, loop will now exit

                except (
                    AllreduceException,
                    MatchmakingException,
                    AssertionError,
                    StopAsyncIteration,
                    asyncio.CancelledError,
                    asyncio.InvalidStateError,
                    P2PHandlerError,
                    P2PDaemonError,
                ) as e:
                    if (
                        step.done()
                        or not step.allow_retries
                        or get_dht_time() >= step.deadline
                    ):
                        if not step.cancelled():
                            hivemind_logger.exception(e)
                        if not step.done():
                            step.set_exception(e)
                    else:
                        hivemind_logger.warning(
                            f"{self.__class__.__name__} caught {repr(e)}, retrying"
                        )

        except BaseException as e:
            if not step.done():
                step.set_exception(e)
            raise
        finally:
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )

    async def _aggregate_with_group(
        self,
        group_info: GroupInfo,
        min_vector_size: int,
        peerids_to_uids: Dict,
        **kwargs,
    ) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(
                *map(self.serializer.loads, group_info.gathered)
            )
            user_gathered = dict(
                zip(
                    group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)
                )
            )
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0
                for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None,
                load_balance_peers,
                self.total_size,
                download_bandwidths,
                min_vector_size,
            )
            bt.logging.info(group_info.peer_ids)
            bt.logging.info(peer_fractions)
            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = DTAllReduceRunner(
                    peerids_to_uids=peerids_to_uids,
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    **kwargs,
                )

                self._running_groups[group_info.group_id].set_result(runner)

                if (
                    runner.modes[group_info.peer_ids.index(self.peer_id)]
                    != AveragingMode.AUX
                ):
                    async for tensor, update in azip(as_aiter(*local_tensors), runner):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()
                else:
                    async for _ in runner:
                        raise ValueError(
                            "aux peers should not receive averaged tensors"
                        )

                return (
                    user_gathered,
                    runner.banned_senders,
                    group_info.peer_ids,
                    modes,
                    download_bandwidths,
                )
        except BaseException as e:
            if isinstance(e, Exception):
                hivemind_logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


class DTGradAverager(DTAverager):
    """
    DTGradAverager is meant to be used in pair with DTStateAverager. Specifically it takes as input the offloaded optimizer of DTStateAverager, and
    use the grad buffer of the offloaded param as averaged_tensors for the DecentralizedAverager. In other words the DTGradAverager makes sure that the grad of the offloaded optimizer
    are kept in sync between peers.

    DTGradAverager is inspired by: https://github.com/PrimeIntellect-ai/OpenDiloco
    """

    def __init__(
        self,
        main_parameters: List[torch.nn.Parameter],
        offloaded_optimizer: TorchOptimizer,
        *,
        dht: DHT,
        prefix: str,
        warn: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ):
        if "averaged_grads" in kwargs:
            raise KeyError(
                "DTGradAverager does not support averaged_grads since it use the offloaded optimizer gradients directly"
            )

        if not isinstance(main_parameters, (list, tuple)):
            raise ValueError(
                "main_parameters must be a list or tuple of torch.nn.Parameter and not an iterator otherwise parameters will be consumed"
            )
        self.main_parameters = list(main_parameters)
        self.offloaded_optimizer = offloaded_optimizer

        self.warn = warn
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0

        self._new_averaged_grads = False

        averaged_grads = tuple(grad for grad in self._grads_from_optimizer())

        self.local_rank = local_rank
        self.world_size = world_size

        super().__init__(
            averaged_tensors=averaged_grads,
            dht=dht,
            prefix=prefix,
            **kwargs,
        )

    def _grads_from_optimizer(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated optimizer"""
        param_groups = self.offloaded_optimizer.param_groups
        for param_group in param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                yield param.grad

    def schedule_step(
        self, scheduled_time: Optional[DHTExpiration] = None, **kwargs
    ) -> StepControl:
        """
        Begin matchmaking: look for a group of peers and prepare for averaging gradients at a specified time.

        :param scheduled_time: expected time when to perform all-reduce. Can be changed using control.scheduled_time
        :param kwargs: any additional keyword args from DecentralizedAverager.step, such as gather, allow_retries, etc
        :note: setting weight at this stage is not supported, please leave this parameter as None
        :returns: step_control - a handle that can be passed into GradientAverager.step to use the pre-scheduled group
        :note: in the current implementation, each step_control can only be used in one step.
        """
        assert (
            kwargs.get("weight") is None
        ), "setting weight in schedule_step is not supported"
        return super().step(
            scheduled_time=scheduled_time, wait=False, require_trigger=True, **kwargs
        )

    def step(
        self,
        control: Optional[StepControl] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        **kwargs,
    ):
        """
        Average accumulated gradients with peers, optionally load averaged gradients and reset accumulators

        :param weight: overrides the averaging weight; by default, weight equals the number of accumulated samples
        :param reset_accumulators: by default, set local gradient accumulators to zeros after averaging succeeds
        :param control: reuse a pre-arranged group of peers (or a matchmaking in progress) from averager.schedule_step
        :param timeout: if specified, await for averaging round for at most this number of seconds (if wait=True)
        :param wait: if True, await for the step to finish (or fail), otherwise run all-reduce in background
        """
        if control is None:
            control = self.schedule_step(timeout=timeout, **kwargs)

        # self.compute_and_load_pseudo_grad_into_averager()
        control.allow_allreduce()

        return control.result(timeout) if wait else control

    @torch.no_grad()
    def compute_and_load_pseudo_grad_into_averager(self):
        """compute pseudo gradient by subtracting the offloaded optimizer parameters with the main parameters and load them in the averager"""
        opt_parameters = [
            param
            for group in self.offloaded_optimizer.param_groups
            for param in group["params"]
        ]
        with self.get_tensors() as averaged_grads:
            for opt_param, averaged_grad, main_param in zip(
                opt_parameters, averaged_grads, self.main_parameters
            ):
                # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                grad = opt_param.data - main_param.detach().to(opt_param.device)
                averaged_grad.copy_(grad, non_blocking=True)

    # worker1, worker2, ... define this
    def get_param_shard(self, idx: int) -> torch.Tensor:
        # idx is the index into self.main_parameters
        return self.main_parameters[idx]

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False

    async def rpc_download_state_partial(
        self, _request: averaging_pb2.DownloadRequest, _context: P2PContext
    ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """
        Get the up-to-date trainer state from a peer.
        The state consists of two parts: (serialized_metadata, tensors)

         - serialized_metadata is a small serialized bytestring meant to store scalars and hyperparameters
         - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        hivemind_logger.info("rpc_download_state_partial")
        if not self.allow_state_sharing:
            return  # deny request and direct peer to the next prospective averager
        metadata, tensors, infos = await self._get_current_state_from_host_process()
        hivemind_logger.info(len(tensors))
        if infos is None:
            infos = [
                CompressionInfo.from_tensor(tensor, key=i)
                for i, tensor in enumerate(tensors)
            ]
        assert len(tensors) == len(infos)

        # for tensor, info in zip([tensors[0]], infos):
        for tensor, info in zip([tensors[0]], infos):
            for part in split_for_streaming(
                self.state_compression.compress(tensor, info, allow_inplace=False)
            ):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(
                        tensor_part=part, metadata=metadata
                    )
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)
            break


@torch.no_grad()
def compute_and_load_pseudo_grad_into_averager(self):
    """
    Rank 0 only:
    - Requests shards via RPC from other ranks one at a time.
    - Reconstructs parameter shard by shard on CPU.
    - Computes pseudo-gradients and loads into averager.
    """
    # assert self.local_rank == 0, "Only rank 0 should call this function!"

    opts = StateDictOptions(
        full_state_dict=True, cpu_offload=True
    )  # gather to CPU on rank 0
    full_state = get_model_state_dict(self.model, options=opts)
    if self.master:
        opt_parameters = [
            p for g in self.outer_optimizer.param_groups for p in g["params"]
        ]
        with self.grad_averager.get_tensors() as averaged_grads:
            for idx, (opt_param, averaged_grad, named_main_param) in enumerate(
                zip(opt_parameters, averaged_grads, full_state.items())
            ):
                _, submod = named_main_param
                # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                grad = opt_param.data - submod.detach().to(opt_param.device)
                averaged_grad.copy_(grad, non_blocking=True)


@torch.no_grad()
def apply_optimizer_parameters(self):
    dist.barrier(group=self.gloo_group)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    opts_get = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    full_state = get_model_state_dict(self.model, options=opts_get)
    self.logger.info("Full state")

    if self.master:
        # Flatten optimizer params in the same order every time
        offloaded_parameters = [
            p for g in self.outer_optimizer.param_groups for p in g["params"]
        ]
        assert len(offloaded_parameters) == len(
            full_state
        ), f"mismatch: {len(offloaded_parameters)} vs {len(full_state)}"
        # full_state values are plain CPU tensors here (no DTensor)
        for (name, tensor), off_t in zip(full_state.items(), offloaded_parameters):
            assert isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
            tensor.copy_(off_t, non_blocking=True)

    opts_set = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=True,
    )
    # Push back into the model (reshard). All ranks must enter.
    set_model_state_dict(
        model=self.model, model_state_dict=full_state, options=opts_set
    )

    del full_state
    gc.collect()
    torch.cuda.empty_cache()
