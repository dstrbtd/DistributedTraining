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

import psutil
import torch
from datetime import datetime
import datetime as dt

from hivemind.utils import get_logger
from huggingface_hub import (
    create_tag,
    hf_hub_download,
    snapshot_download,
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

from torch.distributed._tensor import DeviceMesh
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    set_model_state_dict,
)
import time

hivemind_logger = get_logger(__name__)

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
from itertools import chain
from hivemind.utils import (
    DHTExpiration,
    PerformanceEMA,
    get_logger,
    nested_flatten,
    nested_pack,
)
import bittensor as bt
from packaging.version import Version
import logging

import pickle, hashlib, torch, torch.distributed as dist
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh as DM
from torch.distributed.tensor.placement_types import Shard
import hivemind
from safetensors.torch import save_file, load_file

torch.serialization.add_safe_globals([DTensorSpec])
torch.serialization.add_safe_globals([DTensor])
torch.serialization.add_safe_globals([DM])
torch.serialization.add_safe_globals([Shard])
torch.serialization.add_safe_globals([TensorMeta])

Parameters = Iterable[torch.Tensor]
ParamGroups = Iterable[Dict[str, Any]]
TorchOptimizer = torch.optim.Optimizer
if Version(torch.__version__).major >= 2:
    ZERO_GRAD_SET_TO_NONE_DEFAULT = True
    LRSchedulerBase = torch.optim.lr_scheduler.LRScheduler
else:
    ZERO_GRAD_SET_TO_NONE_DEFAULT = False
    LRSchedulerBase = torch.optim.lr_scheduler._LRScheduler
OptimizerFactory = Callable[[Union[Parameters, ParamGroups]], TorchOptimizer]
SchedulerFactory = Callable[[TorchOptimizer], LRSchedulerBase]


@staticmethod
def check_params(
    optimizer: Union[TorchOptimizer, OptimizerFactory],
    param_groups: Optional[Union[Parameters, ParamGroups]],
    parameter_names: Optional[Sequence[str]],
) -> Tuple[ParamGroups, Sequence[torch.Tensor], Sequence[str]]:
    """Get and verify parameters, groups and names"""
    if param_groups is None:
        assert hasattr(
            optimizer, "param_groups"
        ), "Must provide param_groups or an optimizer with .param_groups"
        param_groups = optimizer.param_groups
    param_groups = tuple(param_groups)
    if all(isinstance(p, torch.Tensor) for p in param_groups):
        param_groups = (dict(params=param_groups),)
    for group in param_groups:
        assert isinstance(group, dict) and group.get("params") is not None
        assert all(isinstance(p, torch.Tensor) for p in group["params"])
    parameters = tuple(chain(*(group["params"] for group in param_groups)))
    if parameter_names is None:
        parameter_names = tuple(i for i in range(len(parameters)))
    parameter_names = tuple(nested_flatten(parameter_names))
    assert len(parameters) == len(
        parameter_names
    ), f"Expected {len(parameters)} names, got {len(parameter_names)}"
    assert len(set(parameters)) == len(
        parameters
    ), "Found duplicate parameters in param_groups"
    params_with_grad = sum(p.numel() for p in parameters if p.requires_grad)
    params_no_grad = sum(p.numel() for p in parameters if not p.requires_grad)
    if params_no_grad >= params_with_grad:
        bt.logging.info(
            "The majority of parameters have requires_grad=False, but they are still synchronized"
            " with peers. If these parameters are frozen (not updated), please do not feed them into "
            "the optimizer at all in order to avoid communication overhead. Proceeding anyway."
        )

    return param_groups, parameters, parameter_names


def make_averaged_parameters(self, main_parameters: Sequence[torch.Tensor]):
    """Initialize averaged parameters based on the optimizer and averaging mode"""
    return tuple(
        make_host_tensor(param, force_copy=self.offload_optimizer)
        for param in main_parameters
    )


def make_host_tensor(
    source_tensor: torch.Tensor, reuse_tensors: bool = False, force_copy: bool = False
) -> torch.Tensor:
    """Create a new tensor for averaging or reuse the existing one"""
    if reuse_tensors and not force_copy:
        if source_tensor.device != torch.device("cpu"):
            raise ValueError(
                "reuse_tensors is only supported if all averaged tensors are on CPU"
            )
        if not source_tensor.is_shared():
            source_tensor.share_memory_()
        return source_tensor
    else:
        averaged_tensor = source_tensor.detach().to(
            device="cpu", dtype=torch.float32, copy=True
        )
        return averaged_tensor.share_memory_().requires_grad_(
            source_tensor.requires_grad
        )


def init_components(
    self,
    main_parameters,
    param_groups: ParamGroups,
    optimizer_or_factory: Union[TorchOptimizer, OptimizerFactory],
    scheduler_or_factory: Optional[Union[LRSchedulerBase, SchedulerFactory]],
    initialize_optimizer: Optional[bool],
) -> Tuple[TorchOptimizer, Optional[LRSchedulerBase]]:
    """Get optimizer and scheduler by either instantiating user-provided factory or using pre-instantiated ones"""
    # assert hasattr(self, "_averaged_parameters"), "Internal error: must initialize averaged parameters first"
    optimizer_is_factory = callable(optimizer_or_factory) and not isinstance(
        optimizer_or_factory, TorchOptimizer
    )
    scheduler_is_factory = callable(scheduler_or_factory) and not isinstance(
        scheduler_or_factory, LRSchedulerBase
    )
    if (
        optimizer_is_factory
        and not scheduler_is_factory
        and scheduler_or_factory is not None
    ):
        raise ValueError(
            "If optimizer is created internally, scheduler must also be initialized internally"
        )
    if self.offload_optimizer and not optimizer_is_factory:
        raise ValueError(
            "Using offload_optimizer requires creating optimizer inside hivemind"
        )
    # self.logger.info("Before make_averaged_parameters")
    averaged_parameters = make_averaged_parameters(self, main_parameters)

    # create optimizer
    if optimizer_is_factory:
        if self.offload_optimizer:
            if self.reuse_tensors:
                parameters_for_optimizer = averaged_parameters
            else:
                parameters_for_optimizer = tuple(
                    tensor.detach().clone().requires_grad_(tensor.requires_grad)
                    for tensor in averaged_parameters
                )

            next_index = 0
            param_groups_for_optimizer = []
            for param_group in param_groups:
                num_params = len(param_group["params"])
                averaged_params_for_group = parameters_for_optimizer[
                    next_index : next_index + num_params
                ]
                param_groups_for_optimizer.append(
                    dict(param_group, params=averaged_params_for_group)
                )
                next_index += num_params
            assert next_index == len(parameters_for_optimizer)

            for param in parameters_for_optimizer:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
        else:
            param_groups_for_optimizer = param_groups
        optimizer = optimizer_or_factory(param_groups_for_optimizer)
    else:
        optimizer = optimizer_or_factory

    # optionally initialize optimizer state dict
    if initialize_optimizer is None:
        initialize_optimizer = not any(
            isinstance(x, torch.Tensor) for x in nested_flatten(optimizer.state_dict())
        )
        bt.logger.info(
            self.status_loglevel,
            "Initializing optimizer manually since it has no tensors in state dict. "
            "To override this, provide initialize_optimizer=False",
        )

    if initialize_optimizer:
        initialize_optimizer_state_(
            optimizer
        )  # note: this will run one optimizer step!

    # create LR scheduler
    if scheduler_is_factory:
        assert callable(scheduler_or_factory)
        scheduler = scheduler_or_factory(optimizer)
    else:
        scheduler = scheduler_or_factory

    # verify optimizer and scheduler
    assert isinstance(optimizer, TorchOptimizer) and len(optimizer.param_groups) == len(
        list(param_groups)
    )
    if self.reuse_tensors:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                assert param.is_shared()
    assert isinstance(scheduler, (LRSchedulerBase, type(None)))
    if scheduler is not None:
        assert scheduler.optimizer == optimizer
    return optimizer, scheduler


def initialize_optimizer_state_(opt: torch.optim.Optimizer):
    """Initialize optimizer statistics by running a virtual optimizer step with zero gradients"""
    flat_params = tuple(
        param for group in opt.param_groups for param in group["params"]
    )
    old_grads = []
    for param in flat_params:
        old_grads.append(param.grad)
        param.grad = torch.zeros_like(param)
    opt.step()
    for param, old_grad in zip(flat_params, old_grads):
        param.grad = old_grad


def check_model_exists(self, repo_id: str, revision: Optional[str] = None) -> bool:
    try:
        if revision and revision != "None":
            list_repo_files(repo_id, revision=revision)
        else:
            list_repo_files(repo_id)
        return True
    except Exception as e:
        self.logger.info(f"Model or revision check failed with error: {e}")
        return False


def _bytes(x):
    return x.numel() * x.element_size()


def summarize_optimizer_state(opt, logger, tag):
    by_dtype = {}
    total = 0
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                nb = _bytes(v)
                total += nb
                key = (str(v.device), str(v.dtype))
                by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] OPT-state total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1])[:12]:
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def summarize_moments_only(opt, logger, tag):
    by_dtype = {}
    total = 0
    for st in opt.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            t = st.get(name, None)
            if isinstance(t, torch.Tensor):
                nb = _bytes(t)
                total += nb
                key = (str(t.device), str(t.dtype))
                by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] MOMENTS total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1]):
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def summarize_grads(model, logger, tag):
    by_dtype = {}
    total = 0
    for p in model.parameters():
        g = p.grad
        if isinstance(g, torch.Tensor):
            nb = _bytes(g)
            total += nb
            key = (str(g.device), str(g.dtype))
            by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] GRADS total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1]):
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def summarize_params(model, logger, tag):
    by_dtype = {}
    total = 0
    for p in model.parameters():
        t = p
        nb = _bytes(t)
        total += nb
        key = (str(t.device), str(t.dtype))
        by_dtype[key] = by_dtype.get(key, 0) + nb
    logger.info(f"[{tag}] PARAMS total = {total/1e9:.3f} GB")
    for (dev, dt), nb in sorted(by_dtype.items(), key=lambda x: -x[1]):
        logger.info(f"[{tag}]  {dev} | {dt} : {nb/1e9:.3f} GB")


def cuda_mem(logger, tag):
    torch.cuda.synchronize()
    logger.info(
        f"[{tag}] alloc={torch.cuda.memory_allocated()/1e9:.3f} GB | "
        f"reserved={torch.cuda.memory_reserved()/1e9:.3f} GB"
    )


def load_model_optimizer_gradient_averager(
    self,
    local_model_name,
    epoch,
    reload_inner_optimizer=True,
    reload_outer_optimizer=True,
    revision=None,
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
        global_model_name, trust_remote_code=True
    )

    if (revision is None) and (local_model_name != global_model_name):
        revision = f"{__run__}.{epoch}.{self.local_progress.inner_step}"
    elif (revision is None) and (local_model_name == global_model_name):
        revision = f"{__run__}.{epoch}.0"

    # # Delete existing outer optimizer
    # if hasattr(self, "outer_optimizer"):
    #     for group in self.outer_optimizer.param_groups:
    #         group["params"].clear()
    #     self.outer_optimizer.state.clear()
    #     del self.outer_optimizer
    #     gc.collect()
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()

    # # Delete existing gradient averager
    # if hasattr(self, "grad_averager"):
    #     self.grad_averager.shutdown()
    #     while self.grad_averager.is_alive():
    #         time.sleep(1)

    #     del self.grad_averager.main_parameters
    #     del self.grad_averager.offloaded_optimizer
    #     del self.grad_averager._averaged_tensors
    #     del self.grad_averager
    #     gc.collect()
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()

    # Delete existing average handler
    if hasattr(self, "avg_handler"):
        del self.avg_handler.model
        del self.avg_handler.inner_optimizer
        del self.avg_handler.grad_averager
        del self.avg_handler
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.logger.info("Deleted Average Handler")

    if hasattr(self, "inner_optimizer"):
        for group in self.inner_optimizer.param_groups:
            group["params"].clear()
        self.inner_optimizer.state.clear()
        del self.inner_optimizer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # if hasattr(self, "model"):
    #     try:
    #         self.model._reset_lazy_init()  # tells FSDP to free params if possible
    #     except Exception:
    #         pass
    #     del self.model
    #     gc.collect()
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()

    loading_success = 0
    optimizer_state = None
    # Load Model & Inner Optimizer
    try:
        if local_model_name == global_model_name:
            revision = ".".join(revision.split(".")[:-1] + ["0"])

        if not check_model_exists(
            self,
            local_model_name,
            revision=revision,
        ):
            raise Exception(f"Failed to load model. Check model exists failed.")

        if not hasattr(self, "model"):
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_name,
                revision=revision,
                trust_remote_code=False,
            )

            need_full_state = self.master and not hasattr(self, "outer_optimizer")
            if need_full_state:
                full_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,  # match your autocast compute dtype
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,  # required by FSDP2 policy
            )

            # Build a 1D device mesh over all ranks
            self.mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))

            # Keep a plain HF module and enable FSDP2 on it
            fully_shard(self.model, mesh=self.mesh, mp_policy=mp_policy)
        else:
            if self.master:
                saftensors_path = hf_hub_download(
                    repo_id=local_model_name,
                    filename="model.safetensors",
                    revision=revision,
                )
                model_state = load_file(saftensors_path, device="cpu")
            else:
                model_state = None

            objs = [model_state]
            dist.broadcast_object_list(objs, src=0, group=self.gloo_group)
            m_blob = objs[0]
            self.logger.info("Downloaded Model State")
            opts = StateDictOptions(
                full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False
            )
            dist.barrier()
            set_model_state_dict(self.model, m_blob, options=opts)

            need_full_state = not hasattr(self, "outer_optimizer")
            self.logger.info("Full state start")
            if need_full_state:
                opts = StateDictOptions(
                    full_state_dict=True,  # gather a full (HF-style) state dict
                    cpu_offload=True,  # offload to host RAM (no GPU OOM)
                )
                full_state = m_blob
            del objs, model_state, m_blob

        self.logger.info(
            f"Successfully Loaded Model From {local_model_name} With Revision {revision}"
        )

        # Move model to device
        self.model.config.block_list = []
        self.local_progress.inner_step = (
            self.model.config.inner_step
            if "inner_step" in self.model.config.__dict__
            else 0
        )
        if (local_model_name == global_model_name) and (
            epoch == self.global_progress.epoch
        ):
            self.allreduce_status_dict = (
                self.model.config.all_reduce_scores
                if "all_reduce_scores" in self.model.config.__dict__
                else {}
            )
        # if reload_inner_optimizer and not hasattr(self, "inner_optimizer"):
        if reload_inner_optimizer:
            if not hasattr(self, "inner_optimizer"):
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
                    repo_id=local_model_name,
                    filename=f"inner_optimizer.rank{self.local_rank+1:04d}-of-{self.world_size}.pt",
                    revision=revision,
                ),
                weights_only=True,
                map_location="cpu",
            )

            opts = StateDictOptions(full_state_dict=False, cpu_offload=True)

            # `optim_sd` is the dict loaded from HF
            self.logger.info(
                f"[r{dist.get_rank()}] before set_optimizer_state_dict (NCCL) {opts}"
            )
            set_optimizer_state_dict(
                model=self.model,
                optimizers=self.inner_optimizer,
                optim_state_dict=optimizer_state["optimizer_state_dict"],
                options=opts,
            )
            self.logger.info(
                f"[r{dist.get_rank()}] after set_optimizer_state_dict (NCCL)"
            )

            if "scheduler_state" in optimizer_state:
                self.scheduler.load_state_dict(optimizer_state["scheduler_state"])

            self.logger.info(
                f"Successfully Loaded Inner Optimizer State From {local_model_name} For Revision {revision}"
            )
            loading_success = 1

    except Exception as e:
        loading_success = 0
        return loading_success
        self.logger.info(
            "local_model_name == global_model_name",
            local_model_name == global_model_name,
        )
        if local_model_name == global_model_name:
            raise Exception(f"Failed to load model despite repo existing: {str(e)}")
        else:
            self.logger.info(f"Failed to load model despite repo existing: {str(e)}")

    finally:
        if isinstance(optimizer_state, dict):
            keys = list(optimizer_state.keys())
            for k in keys:
                del optimizer_state[k]
                gc.collect()
        del optimizer_state
        gc.collect()
        torch.cuda.empty_cache()

    if self.master:
        if not hasattr(self, "outer_optimizer"):
            # Set outer optimizer
            optimizer = partial(torch.optim.SGD, lr=1, momentum=0.9, nesterov=True)

            # param_groups, main_parameters, parameter_names = check_params(optimizer, self.model.parameters(), None)
            param_groups, main_parameters, parameter_names = check_params(
                optimizer, full_state.values(), None
            )
            self.logger.info("Check Params")

            del full_state
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            self.status_loglevel = logging.DEBUG
            self.offload_optimizer = True
            self.custom_gradients = True
            self.reuse_tensors = True
            self.delta_rule_averaging = False
            self._old_tensors: Optional[Sequence[torch.Tensor]] = None  # for delta rule
            scheduler = None
            initialize_optimizer = True

            self.main_parameters, self.parameter_names = (
                main_parameters,
                parameter_names,
            )

            self.outer_optimizer, _ = init_components(
                self,
                main_parameters,
                param_groups,
                optimizer,
                scheduler,
                initialize_optimizer,
            )
            self.logger.info("Init Components")

            # Load a new gradient averager
            self.grad_averager = DTGradAverager(
                dht=self.dht,
                main_parameters=main_parameters,
                offloaded_optimizer=self.outer_optimizer,
                compression=hivemind.Quantile8BitQuantization(),
                prefix=f"{self.config.neuron.run_id}_grad_averager",
                min_group_size=self.config.neuron.min_group_size,
                min_matchmaking_time=30.0,
                request_timeout=10.0,
                next_chunk_timeout=45.0,
                allreduce_timeout=self.allreduce_timeout - 30.0 - 15.0,
                local_rank=self.local_rank,
                world_size=self.world_size,
                start=True,
            )
            self.logger.info("Successfully Loaded Gradient Averager")

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
                    self.outer_optimizer.load_state_dict(
                        optimizer_state["optimizer_state_dict"]
                    )

                self.logger.info(
                    f"Successfully Loaded Outer Optimizer State From {global_model_name} For Revision {'.'.join(revision.split('.')[:-1] + ['0'])}"
                )
                loading_success = 1

            except Exception as e:
                # TODO might need to remove these
                loading_success = 0
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
            self.outer_optimizer,
            self.grad_averager,
            self.retry_limit,
            self.retry_delay,
            self.uid,
            self.config.neuron.local_batch_size_train,
            self.config.neuron.local_batch_size_train_effective,
            self.tokenizer,
            self.device,
            # parameters_list,
        )

        if (
            (self.master)
            and (self.local_progress.inner_step != 0)
            and ("." in revision)
        ):
            self.avg_handler.reset_main_parameters(
                local_model_name,
                revision=".".join(
                    revision.split(".")[:-1]
                    + [
                        str(
                            get_min_local_inner_Step(
                                self, local_model_name, epoch=epoch
                            )
                        )
                    ]
                ),
            )

    self.logger.info(
        f"CPU Memory After Loading State {psutil.virtual_memory().available / 10**9} GB"
    )
    return loading_success


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

        # self.logger.debug("Model Weights Before Loading State")
        # current_model_weights_sample = copy.copy(
        #     [layer for layer in self.model.parameters()][-2][-10:].tolist()
        # )
        # self.logger.debug(current_model_weights_sample)

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
                    loading_success = torch.tensor(
                        [
                            load_model_optimizer_gradient_averager(
                                self,
                                local_model_name=repo_id,
                                epoch=epoch,
                                reload_inner_optimizer=reload_inner_optimizer,
                                reload_outer_optimizer=reload_outer_optimizer,
                                revision=revision,
                            )
                        ]
                    )
                    dist.all_reduce(loading_success, group=self.gloo_group)
                    if (loading_success[0].item() != self.world_size) and (
                        repo_id != self.config.neuron.global_model_name
                    ):
                        self.logger.info(
                            "LOAD GLOBAL MODEL",
                            load_model_optimizer_gradient_averager(
                                self,
                                self.config.neuron.global_model_name,
                                self.global_progress.epoch,
                            ),
                        )
                    break

                except Exception as e:
                    attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        if use_fallback_model:
                            # TODO Crash the whole process if global model is not loaded
                            loading_success = torch.tensor(
                                [
                                    load_model_optimizer_gradient_averager(
                                        self,
                                        self.config.neuron.global_model_name,
                                        self.global_progress.epoch,
                                    ),
                                ]
                            )
                        else:
                            raise Exception(
                                f"Failed to load model after {MAX_ATTEMPTS} attempts: {str(e)}"
                            )
                    self.logger.info(
                        f"Failed to load model, retrying. Attempt {attempt}/{MAX_ATTEMPTS}. Error {str(e)}"
                    )
                    self.logger.info(e)

            state_loaded = True

            # self.logger.debug("Model Weights After Loading State")
            # new_model_weights_sample = copy.copy(
            #     [layer for layer in self.model.parameters()][-2][-10:].tolist()
            # )
            # self.logger.debug(new_model_weights_sample)

            self.local_progress.epoch = epoch
            self.local_progress.samples_accumulated = 0
            self.logger.info(f"New Model Tag: {self.global_progress.epoch}")

            # Clean up old cache
            try:
                cleanup_old_cache(self, repo_id, revision)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup cache: {str(e)}")

            if repo_id != self.config.neuron.global_model_name:
                try:
                    cleanup_old_cache(
                        self,
                        self.config.neuron.global_model_name,
                        current_revision=None,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup cache: {str(e)}")

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


def upload_new_state(
    self,
    epoch: int,
    results: dict,
    model_state: dict,
    inner_optimizer_state: dict,
    inner_optimizer_lr: int,
    block: int = None,
):
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            self.logger.info(
                f"Pushing new model and optimizer state to HF Hub with tag {epoch}"
            )

            # Save and upload both model and optimizer state
            upload_success = save_and_upload_state(
                self,
                epoch=epoch,
                results=results,
                model_state=model_state,
                inner_optimizer_state=inner_optimizer_state,
                inner_optimizer_lr=inner_optimizer_lr,
                block=block,
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


def save_and_upload_state(
    self,
    epoch: int,
    results: dict,
    model_state: dict,
    inner_optimizer_state: dict,
    inner_optimizer_lr: int,
    block: int = None,
):
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
                self.model.config.save_pretrained(tmp_folder)
                self.logger.info("Save config")
                save_file(
                    model_state,
                    os.path.join(tmp_folder, "model.safetensors"),
                    metadata={"format": "pt"},
                )
                self.logger.info("Save model")

                # Save outer optimizer state
                outer_optimizer_state = {
                    "optimizer_state_dict": self.outer_optimizer.state_dict(),
                    "learning_rate": self.outer_optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
                torch.save(
                    outer_optimizer_state,
                    os.path.join(tmp_folder, "outer_optimizer.pt"),
                )
                self.logger.info("Save optimizer")
                # TODO Save non sharded inner optimizer
                # Save outer optimizer state
                inner_optimizer_state = {
                    "optimizer_state_dict": inner_optimizer_state,
                    "learning_rate": inner_optimizer_lr,
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
