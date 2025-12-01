import json
import torch.distributed as dist

from dataclasses import dataclass
from distributed_training import __run__
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint


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
    score: float = 0.0


class Chaindata(BaseModel):
    last_updated_block: int = 0


class ScoreAllReduce(BaseModel):
    peer_id: Optional[str] = None
    score: float = 0.0
    count: int = 0


class ScoreTrain(BaseModel):
    r2_hash: Optional[str] = None
    model_id: Optional[str] = None
    account_id: Optional[str] = "x" * 32
    access_key_id: Optional[str] = "x" * 32
    secret_access_key: Optional[str] = "x" * 64
    is_valid: bool = True
    random: LossProfile = Field(default_factory=LossProfile)
    assigned: LossProfile = Field(default_factory=LossProfile)
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


def get_r2_client(self, uid: int, donwload_on_all_ranks: bool):
    if uid == self.uid:
        account_id = self.config.r2.account_id
        access_key_id = self.config.r2.read.access_key_id
        secret_access_key = self.config.r2.read.secret_access_key
    elif uid == self.master_uid:
        return self.r2["global"]
    elif donwload_on_all_ranks:
        if self.master:
            account_id = self.uid_tracker[uid].train.account_id
            access_key_id = self.uid_tracker[uid].train.access_key_id
            secret_access_key = self.uid_tracker[uid].train.secret_access_key
        else:
            account_id = self.config.r2.account_id
            access_key_id = self.config.r2.read.access_key_id
            secret_access_key = self.config.r2.read.secret_access_key
        commitment = [account_id + access_key_id + secret_access_key]
        dist.broadcast_object_list(commitment, src=0, group=self.gloo_group)
        self.logger.debug(f"UID {uid:03d}: Commitment - {commitment}")
        account_id = commitment[0][:32]
        access_key_id = commitment[0][32:64]
        secret_access_key = commitment[0][64:]
    else:
        account_id = self.uid_tracker[uid].train.account_id
        access_key_id = self.uid_tracker[uid].train.access_key_id
        secret_access_key = self.uid_tracker[uid].train.secret_access_key

    self.logger.debug(account_id, access_key_id, secret_access_key)

    if (
        (account_id == "x" * 32)
        or (access_key_id == "x" * 32)
        or (secret_access_key == "x" * 64)
    ):
        raise ValueError(f"UID {uid:03d} has no R2 credentials.")

    return self.session.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )


def get_progress(
    self,
    local_or_global: str,
    bucket_name: str = None,
    uid: int = None,
    epoch: int = None,
    donwload_on_all_ranks=True,
):
    # local_or_global is used for miners
    # uid is used for validators to cycle through progress of different uids
    if (local_or_global != "global") and (bucket_name is None) and (uid is None):
        bucket_name = self.config.r2.bucket_name
    elif (local_or_global != "global") and (uid == self.master_uid):
        bucket_name = f"{self.config.neuron.global_model_name}-{uid:03d}"
    elif (uid is not None) and (uid != self.master_uid):
        bucket_name = f"{self.config.neuron.global_model_name}-{uid:03d}"
    elif (local_or_global == "global") or (uid == self.master_uid):
        bucket_name = self.config.neuron.global_model_name

    try:
        if uid is not None:
            r2 = get_r2_client(self, uid, donwload_on_all_ranks)
        else:
            r2 = self.r2[local_or_global]

        obj = r2.get_object(Bucket=bucket_name, Key="metadata.json")
        data = obj["Body"].read()
        metadata = json.loads(data)
        local_epoch = metadata["outer_step"]
        local_inner_step = metadata["inner_step"]
        local_peer_id = metadata["peer_id"]
        return local_epoch, local_inner_step, local_peer_id
    except Exception as e:
        self.logger.debug(f"Error in get_progress: {str(e)}")
        return None, 0, None
