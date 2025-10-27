# upload_worker.py
import sys
import pathlib
import boto3
from botocore.config import Config

from distributed_training.utils.r2 import (
    upload_folder_to_r2,
    archive_root_bucket,
    restore_from_epoch,
)

if __name__ == "__main__":
    bucket = sys.argv[1]
    r2_account_id = sys.argv[2]
    r2_write_access_access_key_id = sys.argv[3]
    r2_write_access_secret_access_key = sys.argv[4]
    tag = sys.argv[5]
    archive = sys.argv[6]
    epoch = tag.split(".")[1]
    prefix = f"epoch-{epoch}/"
    restore = "True"

    r2_write = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_write_access_access_key_id,
        aws_secret_access_key=r2_write_access_secret_access_key,
        region_name="auto",
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},  # or "standard"
            connect_timeout=30,
            read_timeout=120,
            max_pool_connections=50,
        ),
    )

    upload_folder_to_r2(r2_write, bucket, prefix)
    # Only archive on the miner side after an AllReduce
    # Variable has to be fed as a string in subprocess
    if archive == "True":
        archive_root_bucket(r2_write, bucket, epoch)

    # if restore == "True":
    #     restore_from_epoch(r2_write, bucket, epoch)

    # r2_write.upload_file(
    #     str("/root/llama-4b-ws-4/metadata.json"), bucket, f"metadata.json"
    # )
