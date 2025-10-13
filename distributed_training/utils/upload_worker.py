# upload_worker.py
import sys
import boto3
import pathlib
import datetime

import tqdm
import os
import boto3, pathlib
from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor
from distributed_training.utils.state_loader import archive_root_bucket


def upload_folder_to_r2(r2, r2_bucket_id, prefix="", max_workers=14):
    local_folder = pathlib.Path(r2_bucket_id)

    files = [p for p in local_folder.rglob("*") if p.is_file()]
    print(f"Uploading {len(files)} files with {max_workers} threads...")

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

        print(str(path))
        print(key)
        r2.upload_file(str(path), r2_bucket_id, key, Config=cfg)
        pbar.update(1)
        return key

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, key in enumerate(ex.map(_upload, files), 1):
            if i % 10 == 0 or i == len(files):
                print(f"Uploaded {i}/{len(files)}")

    print("✅ Upload complete")
    print(datetime.datetime.now())


if __name__ == "__main__":
    r2_bucket_id = sys.argv[1]
    r2_account_id = sys.argv[2]
    r2_write_access_access_key_id = sys.argv[3]
    r2_write_access_secret_access_key = sys.argv[4]
    tag = sys.argv[5]
    epoch = tag.split(".")[1]

    r2_write = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_write_access_access_key_id,
        aws_secret_access_key=r2_write_access_secret_access_key,
        region_name="auto",
    )

    upload_folder_to_r2(r2_write, r2_bucket_id)
    archive_root_bucket(r2_write, r2_bucket_id, epoch)

    # local_folder = pathlib.Path(f"{r2_bucket_id}/metadata.json")
    # r2_write.upload_file(str(local_folder), r2_bucket_id, f"metadata.json")
