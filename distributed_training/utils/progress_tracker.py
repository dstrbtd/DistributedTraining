from dataclasses import dataclass
from distributed_training import __run__

import bittensor as bt
import pandas as pd
from huggingface_hub import list_repo_refs
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint
from tqdm import tqdm

import wandb


@dataclass(frozen=False)
class GlobalTrainingProgress:
    epoch: int
    samples_accumulated: int


class LocalTrainingProgress(BaseModel):
    peer_id: bytes
    epoch: conint(ge=0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    time: StrictFloat
    client_mode: StrictBool
    inner_step: conint(ge=0, strict=True)
    loss: confloat(ge=0.0, strict=True)


from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SkillRating(BaseModel):
    mu: float
    sigma: float


class LossProfile(BaseModel):
    before: float = 0.0
    after: float = 0.0
    absolute: float = 0.0
    relative: float = 0.0


class Chaindata(BaseModel):
    last_updated_block: int = 0


class ScoreAllReduce(BaseModel):
    peer_id: Optional[str] = None
    score: float = 0.0
    count: int = 0


class ScoreTrain(BaseModel):
    model_id: Optional[str] = None
    is_valid: bool = True
    loss: LossProfile = Field(default_factory=LossProfile)
    openskill_rating: float = 0.0
    score: float = 0.0
    updated_time: float = 0
    revision: str = "0.0.0"
    openskill_rating: SkillRating = Field(
        default_factory=lambda: SkillRating(mu=25.0, sigma=8.333)
    )


class ScoreTotal(BaseModel):
    score: float = 0.0


class UidTracker(BaseModel):
    uid: int
    all_reduce: ScoreAllReduce = Field(default_factory=ScoreAllReduce)
    train: ScoreTrain = Field(default_factory=ScoreTrain)
    total: ScoreTotal = Field(default_factory=ScoreTotal)
    chaindata: Chaindata = Field(default_factory=Chaindata)


def get_global_epoch(self):
    try:
        refs = list_repo_refs(self.config.neuron.global_model_name, repo_type="model")
        global_epoch = (
            max(
                [
                    int(tag.name.split(".")[1])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                    )
                ]
            )
            if refs.tags
            else 0
        )
        return global_epoch
    except Exception as e:
        bt.logging.warning(f"Error in get_global_epoch: {str(e)}")
        return 0


def get_local_epoch(self, repo_id: str = None):
    if repo_id is None:
        repo_id = self.config.neuron.local_model_name

    try:
        refs = list_repo_refs(repo_id, repo_type="model")
        local_epoch = (
            max(
                [
                    int(tag.name.split(".")[1])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                    )
                ]
            )
            if refs.tags
            else None
        )
        return local_epoch
    except Exception as e:
        bt.logging.warning(f"Error in get_local_epoch: {str(e)}")
        return None


def get_local_inner_step(self, repo_id: str = None, epoch: int = None):
    if repo_id is None:
        repo_id = self.config.neuron.local_model_name
    if epoch is None:
        epoch = self.local_progress.epoch

    try:
        refs = list_repo_refs(repo_id, repo_type="model")
        local_steps = (
            max(
                [
                    int(tag.name.split(".")[2])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                        and (tag.name.split(".")[1] == str(epoch))
                    )
                ]
            )
            if refs.tags
            else 0
        )
        return local_steps
    except Exception as e:
        bt.logging.warning(f"Error in get_local_inner_step: {str(e)}")
        return 0


def get_min_local_inner_Step(self, repo_id: str = None, epoch: int = None):
    if repo_id is None:
        repo_id = self.config.neuron.local_model_name
    if epoch is None:
        epoch = self.local_progress.epoch

    try:
        refs = list_repo_refs(repo_id, repo_type="model")
        local_steps = (
            min(
                [
                    int(tag.name.split(".")[2])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                        and (tag.name.split(".")[1] == str(epoch))
                    )
                ]
            )
            if refs.tags
            else 0
        )
        return local_steps
    except Exception as e:
        bt.logging.warning(f"Error in get_local_inner_step: {str(e)}")
        return 0
