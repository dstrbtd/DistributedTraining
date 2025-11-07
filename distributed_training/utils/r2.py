import torch
import tqdm
import os
import tempfile
import filelock
import datetime
import pathlib
import json
import torch.distributed as dist

from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor
from botocore.client import BaseClient
from s3transfer.manager import TransferManager
from boto3.s3.transfer import TransferConfig
from distributed_training import __run__

ACCEPTED_FILES = [
    "config.json",
    "model.safetensors",
    "gradients.pt",
    "inner_optimizer.rank0001-of-4.pt",
    "inner_optimizer.rank0002-of-4.pt",
    "inner_optimizer.rank0003-of-4.pt",
    "inner_optimizer.rank0004-of-4.pt",
    "outer_optimizer.pt",
]


def upload_folder_to_r2(r2, bucket, prefix="", max_workers=8):
    local_folder = pathlib.Path(bucket)

    files = [p for p in local_folder.rglob("*") if p.is_file()]
    # print(f"Uploading {len(files)} files with {max_workers} threads...")

    pbar = tqdm.tqdm(total=len(files))

    def _upload(path):
        key = f"{prefix}{path.relative_to(local_folder)}"
        print(key)
        # key = str(path.relative_to(local_folder))
        size = os.path.getsize(path)

        if key.split("/")[-1] not in ACCEPTED_FILES:
            return key

        if size > 512:
            threshold = 64
            workers = 12
        else:
            threshold = 32
            workers = 8

        if size < 512 * 1024**2:  # < 512 MB
            threshold, workers = 16, 6
        elif size < 2 * 1024**3:  # < 2 GB
            threshold, workers = 32, 10
        else:  # > 2 GB
            threshold, workers = 64, 14

        cfg = TransferConfig(
            multipart_threshold=threshold * 1024 * 1024,  # 8 MB
            multipart_chunksize=threshold * 1024 * 1024,
            max_concurrency=workers,
            use_threads=True,
        )
        r2.upload_file(str(path), bucket, key, Config=cfg)
        pbar.update(1)
        return key

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, key in enumerate(ex.map(_upload, files), 1):
            if i % 10 == 0 or i == len(files):
                pass
                # print(f"Uploaded {i}/{len(files)}")

    # print("✅ Upload complete")
    # print(datetime.datetime.now())


def archive_root_bucket(r2: BaseClient, bucket: str, epoch: int):
    print("⌛️ Archive start")
    print(datetime.datetime.now())
    # multipart thresholds/chunks; tune as needed
    tcfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,  # 8MB
        multipart_chunksize=64 * 1024 * 1024,  # 64MB
        max_concurrency=4,
        use_threads=True,
    )

    archive_prefix = f"epoch-{epoch}/"
    paginator = r2.get_paginator("list_objects_v2")
    with TransferManager(r2, config=tcfg) as tm:
        futures = []
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # ✅ skip pseudo-folders or empty keys
                if (
                    (not key)
                    or ("epoch-" in key)
                    or (obj["Size"] == 0)
                    or (key not in ACCEPTED_FILES)
                ):
                    continue

                dest_key = f"{archive_prefix}{key}"

                futures.append(
                    tm.copy(
                        copy_source={"Bucket": bucket, "Key": key},
                        bucket=bucket,
                        key=dest_key,
                        extra_args={"MetadataDirective": "COPY"},
                    )
                )

        # wait for all copies to finish (raises on failure)
        for f in futures:
            f.result()
    r2.close()
    print("✅ Archive complete")
    print(datetime.datetime.now())


def restore_from_epoch(r2: BaseClient, bucket: str, epoch: int):
    """
    Copies all objects from epoch-{epoch}/ back into the main bucket root.
    """
    tcfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,  # 8MB
        multipart_chunksize=64 * 1024 * 1024,  # 64MB
        max_concurrency=4,
        use_threads=True,
    )

    source_prefix = f"epoch-{epoch}/"
    paginator = r2.get_paginator("list_objects_v2")

    with TransferManager(r2, config=tcfg) as tm:
        futures = []
        for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # skip empty or malformed entries
                if not key or obj["Size"] == 0:
                    continue

                # remove the epoch prefix so files go to root
                dest_key = key[len(source_prefix) :]

                # skip if key would become empty (folder marker)
                if not dest_key:
                    continue

                futures.append(
                    tm.copy(
                        copy_source={"Bucket": bucket, "Key": key},
                        bucket=bucket,
                        key=dest_key,
                        extra_args={"MetadataDirective": "COPY"},
                    )
                )

        for f in futures:
            f.result()

    r2.close()


def r2_download(
    self,
    r2,
    bucket,
    key,
    donwload_on_all_ranks=True,
    run_on_all_ranks=True,
    destination=None,
):
    if destination is None:
        fd, destination_path = tempfile.mkstemp()
        os.close(fd)
    else:
        destination_path = destination
        destination_path = os.path.join(
            destination_path, os.path.basename(key.split("/")[-1])
        )

    # Let only the master perform the actual download
    if (self.master) or (donwload_on_all_ranks):
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            lock_path = destination_path + ".lock"
            with filelock.FileLock(lock_path):
                r2.download_file(bucket, key, destination_path)
            success = torch.tensor([1], dtype=torch.int, device="cuda")
        except Exception as e:
            self.logger.info(f"Download failed due to error: {e}")
            success = torch.tensor([0], dtype=torch.int, device="cuda")
    else:
        success = torch.tensor([0], dtype=torch.int, device="cuda")

    if donwload_on_all_ranks or run_on_all_ranks:
        # Broadcast success flag from master to everyone
        dist.broadcast(success, src=0)

        # If master failed, all ranks raise the same error
        if success.item() == 0:
            raise RuntimeError("Master rank failed during r2_download().")

    return destination_path


def log_peerid_to_r2(self, prefix=""):
    if self.master:
        # Save metadata
        metadata = {
            "run": int(__run__),
            "outer_step": int(self.local_progress.epoch),
            "inner_step": int(self.local_progress.inner_step),
            "peer_id": str(self.dht.peer_id.to_base58()),
        }
        with open(os.path.join(self.output_dir, f"metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
        # Upload Peer Metadata With Updated Peer ID
        self.r2["write"].upload_file(
            str(os.path.join(self.output_dir, "metadata.json")),
            self.config.r2.bucket_name,
            f"{prefix}metadata.json",
        )
