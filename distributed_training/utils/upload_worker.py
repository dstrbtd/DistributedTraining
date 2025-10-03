# upload_worker.py
import sys
import boto3
from huggingface_hub import upload_folder
import pathlib


def upload_folder_to_r2(local_folder, bucket, prefix=""):
    local_folder = pathlib.Path(local_folder)
    for path in local_folder.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_folder)}"
            print(f"Uploading {path} to s3://{bucket}/{key}")
            r2.upload_file(str(path), bucket, key)


if __name__ == "__main__":
    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    commit_message = sys.argv[3]

    R2_DATASET_ACCOUNT_ID = "b8ef6a12e4c28b77ecda126d6f02ab07"
    R2_DATASET_READ_ACCESS_KEY_ID = "c838304b4d1d11a387dc2220bbe6bd8b"
    R2_DATASET_READ_SECRET_ACCESS_KEY = (
        "59076c96a54e53b35f7d61d4feef32881e005c5a9d18371e0977c40432118384"
    )

    r2 = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_DATASET_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_DATASET_READ_ACCESS_KEY_ID,
        aws_secret_access_key=R2_DATASET_READ_SECRET_ACCESS_KEY,
        region_name="auto",
    )

    upload_folder_to_r2(local_dir, repo_id, prefix="checkpoints/run-0.0.0")
