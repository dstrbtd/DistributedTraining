# upload_worker.py
import sys
import boto3
import pathlib
import datetime
import tqdm
import os

from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor
from botocore.client import BaseClient
from s3transfer.manager import TransferManager
from boto3.s3.transfer import TransferConfig


def upload_folder_to_r2(r2, bucket, prefix="", max_workers=14):
    local_folder = pathlib.Path(bucket)

    files = [p for p in local_folder.rglob("*") if p.is_file()]
    # print(f"Uploading {len(files)} files with {max_workers} threads...")

    pbar = tqdm.tqdm(total=len(files))

    def _upload(path):
        key = f"{prefix}/{path.relative_to(local_folder)}"
        key = str(path.relative_to(local_folder))
        size = os.path.getsize(path)

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
                if (not key) or ("epoch-" in key) or (obj["Size"] == 0):
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


if __name__ == "__main__":
    bucket = sys.argv[1]
    r2_account_id = sys.argv[2]
    r2_write_access_access_key_id = sys.argv[3]
    r2_write_access_secret_access_key = sys.argv[4]
    tag = sys.argv[5]
    archive = sys.argv[6]
    epoch = tag.split(".")[1]

    r2_write = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_write_access_access_key_id,
        aws_secret_access_key=r2_write_access_secret_access_key,
        region_name="auto",
    )

    upload_folder_to_r2(r2_write, bucket)
    # Only archive on the miner side after an AllReduce
    if archive:
        archive_root_bucket(r2_write, bucket, epoch)

    # local_folder = pathlib.Path(f"{bucket}/metadata.json")
    # r2_write.upload_file(
    #     str("/root/llama-1b-ws-2/metadata.json"), bucket, f"metadata.json"
    # )
