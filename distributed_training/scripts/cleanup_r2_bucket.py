import os
import time

import boto3
from dotenv import load_dotenv

load_dotenv()

bucket = "llama-1b-ws-4-000"

r2 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ADMIN_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_ADMIN_SECRET_ACCESS_KEY"],
    region_name="auto",
)


# 1️⃣  Delete all objects
def delete_all_objects(bucket):
    print("Deleting objects...")
    paginator = r2.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objs = page.get("Contents", [])
        if not objs:
            continue
        to_delete = [{"Key": o["Key"]} for o in objs]
        r2.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
        print(f"Deleted {len(to_delete)} objects")
    print("✅ All objects deleted")


# 2️⃣  Abort any ongoing multipart uploads
def abort_all_multipart(bucket):
    print("Aborting multipart uploads...")
    while True:
        resp = r2.list_multipart_uploads(Bucket=bucket)
        uploads = resp.get("Uploads", [])
        if not uploads:
            break
        for u in uploads:
            r2.abort_multipart_upload(
                Bucket=bucket, Key=u["Key"], UploadId=u["UploadId"]
            )
        print(f"Aborted {len(uploads)} uploads")
        # small delay to let R2 finalize the aborts
        time.sleep(0.5)
    print("✅ All multipart uploads aborted")


# 3️⃣  Now delete the bucket itself
def delete_bucket(bucket):
    print("Deleting bucket...")
    r2.delete_bucket(Bucket=bucket)
    print("✅ Bucket deleted")


# Execute
delete_all_objects(bucket)
abort_all_multipart(bucket)
delete_bucket(bucket)
