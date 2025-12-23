import asyncio
import time
import random
import datetime
from zoneinfo import ZoneInfo
from typing import Callable, Dict

from dotenv import load_dotenv
import boto3
import os
import gc
import sys
import torch
import shutil
import json
import torch.distributed as dist
from distributed_training import __run__
from distributed_training.data.dataset import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from pathlib import Path
from influxdb_client import InfluxDBClient, Point, WritePrecision
from tabulate import tabulate
from torch.distributed._tensor import DeviceMesh
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
from botocore.config import Config
from distributed_training.utils.r2 import (
    upload_folder_to_r2,
    r2_download,
    log_peerid_to_r2,
)
import logging

load_dotenv()
# === CONFIG ===
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")
INFLUXDB_MEASUREMENT = "evaluation_metrics"
BUCKET = "llama-4b-ws-4"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_SKIP_PROBABILITY = 0.9
EVAL_DURATION_MINUTES = 15
EVAL_TYPES = ["fineweb", "lm_eval"]  # Add lm-eval harness tasks in the future
R2 = boto3.session.Session().client(
    "s3",
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv("R2_READ_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_READ_SECRET_ACCESS_KEY"),
    region_name="auto",
    config=Config(
        retries={"max_attempts": 10, "mode": "adaptive"},  # or "standard"
        connect_timeout=30,
        read_timeout=120,
        max_pool_connections=50,
    ),
)
__run__ = "4"

# === LOGGER SETUP ===
logging.basicConfig(
    level=logging.INFO,  # allow INFO through
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
    force=True,  # override any prior config
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setLevel(logging.INFO)
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(h)
logger.propagate = False

# === INFLUXDB SETUP ===
influx = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx.write_api()
query_api = influx.query_api()
delete_api = influx.delete_api()


class Dummy:
    def __init__(self):
        # Set distributed variables
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.master = self.local_rank == 0

        if not dist.is_initialized():
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="tcp://127.0.0.1:29500",
                    rank=self.local_rank,
                    world_size=self.world_size,
                )
            if not hasattr(self, "gloo_group"):
                self.gloo_group = dist.new_group(
                    backend="gloo",
                )
        self.logger = logger


SELF = Dummy()


def tag_exists(tag: str, task: str) -> bool:
    if task == "lm_eval":
        task = "mmlu_stem.acc"
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: -365d)
    |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.tag == "{tag}" and r.task == "{task}")
    |> limit(n:1)
    """
    result = query_api.query(org=INFLUXDB_ORG, query=query)
    return len(result) > 0


def log_score(
    tag: str,
    task: str,
    score: float,
    output_dir: str = None,
):
    if task == "fineweb":
        point = (
            Point(INFLUXDB_MEASUREMENT)
            .tag("tag", tag)
            .tag("task", task)
            .field("score", score)
            .time(datetime.datetime.now(datetime.timezone.utc), WritePrecision.NS)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    else:
        directory = f"{os.getcwd()}/{output_dir}/{BUCKET.replace('/', '__')}"
        json_file = f"{directory}/{os.listdir(directory)[0]}"
        new_output_dir = (
            f"{os.path.dirname(os.path.abspath(__file__))}/{output_dir}.json"
        )

        # Load JSON
        with open(json_file, "r") as f:
            data = json.load(f)

        timestamp = int(data.get("date", 0) * 1e9)  # Influx expects ns
        for task, values in data["results"].items():
            for metric, score in values.items():
                if metric == "alias":
                    continue  # skip alias itself
                # else:
                #     print(task+"."+metric.replace(",none", ""), score)
                try:
                    score = float(score)
                    point = (
                        Point(INFLUXDB_MEASUREMENT)  # measurement
                        .tag("tag", tag)
                        .tag("task", f"{task}.{metric.replace(',none', '')}")
                        .field("score", score)
                        .time(timestamp, WritePrecision.NS)
                    )

                    write_api.write(
                        bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point
                    )

                except (TypeError, ValueError) as e:
                    print(f"An error occurred: {e}")
                    continue  # skip non-numeric

        # ---- PRINT SUMMARY TABLE ----
        results = data["results"]

        def get_score(task, metric):
            val = results.get(task, {}).get(metric, None)
            return f"{val*100:.1f}" if isinstance(val, float) else "N/A"  # convert to %

        # Collect rows
        rows = [
            [
                "DSTRBTD-1.10B",
                "FineWebEdu",
                f'~{int(80*int(tag.split(".")[1])*100*512*1025/1e9)}B',  # 32 peers per outer step, 65 outer steps, 100 inner steps, 512 samples per inner_step, 1024 Tokens Per Sample
                get_score("hellaswag", "acc_norm,none"),
                get_score("piqa", "acc_norm,none"),
                get_score("arc_easy", "acc,none"),
            ],
            [
                "TEMPLAR-1.21B",
                "FineWebEdu",
                "100B-200B",
                51.0,
                71.4,
                59.2,
            ],
            [
                "DEM0-1.18B",
                "Dolmo",
                "100B",
                "48.0",
                "71.0",
                "55.0",
            ],
            [
                "DILOCO-1.30B",
                "Dolmo",
                "26B",
                "45.0",
                "68.4",
                "39.0",
            ],
        ]

        print(
            tabulate(
                rows,
                headers=[
                    "Model",
                    "Dataset",
                    "Tokens",
                    "HellaSwag acc_norm",
                    "PIQA acc_norm",
                    "ARC-E acc",
                ],
                tablefmt="fancy_grid",
            )
        )

        shutil.rmtree(f"{os.getcwd()}/{output_dir}")
        with open(new_output_dir, "w") as f:
            json.dump(data, f, indent=4)  # indent=4 for pretty printing


async def fetch_training_data(tokenizer):
    """Async function to fetch training data"""
    retry_limit = 10
    retry_delay = 60
    attempt = 0
    local_batch_size_train = 4
    if dist.get_rank() == 0:
        current_block = random.randint(6193881 * 2, 6193881 * 4)
        uid = random.randint(300, 1000000)
        tensor = torch.tensor([current_block, uid], dtype=torch.long, device="cuda")
    else:
        tensor = torch.zeros(2, dtype=torch.long, device="cuda")

    # Broadcast from rank 0 to all others
    dist.broadcast(tensor, src=0)
    current_block = int(tensor[0].item())
    uid = int(tensor[1].item())
    # print(SELF.local_rank, f"Fetched block {current_block} with uid {uid}")
    while attempt < retry_limit:
        try:
            pages = await DatasetLoader.next_pages(
                offset=current_block,
                n_pages=5,
                seed=uid,
            )
            random.seed(uid)
            random.shuffle(pages)

            dataset = await DatasetLoader.create(
                batch_size=local_batch_size_train,
                sequence_length=1024,
                pages_info=pages,
                tokenizer=tokenizer,
            )

            dataset_length = torch.tensor(len(dataset.buffer))
            dist.all_reduce(dataset_length, op=dist.ReduceOp.MIN, group=SELF.gloo_group)
            dataset.buffer = dataset.buffer[:dataset_length]

            return dataset
        except Exception as e:
            print(f"Error fetching training data: {str(e)}")
            attempt += 1
            print(f"Failed to fetch data, retrying. Attempt {attempt}/{retry_limit}")
            if attempt < retry_limit:
                time.sleep(retry_delay * attempt)  # Wait before the next retry
            else:
                print("Maximum retry limit reached. Unable to fetch data.")
                raise


# === EVALUATORS ===
def evaluate_fineweb(
    device: str,
    tag: str,
    max_minutes: int = EVAL_DURATION_MINUTES,
) -> float:
    """
    Stream and evaluate a fixed-time sample of fineweb-edu on average LM loss.

    Args:
        model: HuggingFace model
        tokenizer: Matching tokenizer
        device: cuda or cpu
        max_minutes: Time budget in minutes
        max_seq_length: Max input length for tokenization

    Returns:
        Average loss
    """
    prefix = f"epoch-{tag.split('.')[1]}/"
    output_dir = os.path.join(os.getcwd(), BUCKET)
    _ = r2_download(
        SELF,
        r2=R2,
        bucket=BUCKET,
        key=f"{prefix}model.safetensors",
        donwload_on_all_ranks=False,
        destination=output_dir,
    )
    _ = r2_download(
        SELF,
        r2=R2,
        bucket=BUCKET,
        key=f"{prefix}config.json",
        donwload_on_all_ranks=False,
        destination=output_dir,
    )
    dist.barrier(device_ids=[SELF.local_rank])
    tokenizer = AutoTokenizer.from_pretrained("dstrbtd/llama-1b")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,  # match your autocast compute dtype
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,  # required by FSDP2 policy
    )

    # Build a 1D device mesh over all ranks
    mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))
    # Keep a plain HF module and enable FSDP2 on it
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    model.eval()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    loop = asyncio.new_event_loop()

    while (time.time() - start_time) <= (max_minutes * 30):
        # Use streaming mode
        dataset = loop.run_until_complete(fetch_training_data(tokenizer))

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                if random.random() > (1 - DATASET_SKIP_PROBABILITY):
                    continue

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                if inputs is None or len(inputs) == 0:
                    print(f"Empty batch at index {i}, skipping")
                    continue

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=inputs, labels=inputs)
                    total_loss += outputs.loss.item()
                    n_batches += 1
                    if n_batches % 20 == 0 and SELF.master:
                        SELF.logger.info(total_loss / n_batches)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # local aggregates
    local_loss = torch.tensor([total_loss], dtype=torch.float64, device="cuda")
    local_count = torch.tensor([n_batches], dtype=torch.int64, device="cuda")

    logger.info(f"{SELF.local_rank},{local_loss},{local_count}")
    # sum across all ranks (in-place; now identical on every rank)
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    logger.info(f"{SELF.local_rank},{local_loss},{local_count}")
    global_total_loss = float(local_loss.item())
    global_n_batches = int(local_count.item())

    score = (
        (global_total_loss / global_n_batches) if global_n_batches > 0 else float("inf")
    )
    logger.info(f"{SELF.local_rank},{score}")

    if SELF.master:
        log_score(tag, "fineweb", score)
    dist.barrier(device_ids=[SELF.local_rank])
    return score


def evaluate_with_lm_harness(
    device: str,
    tag: str,
) -> float:
    """
    Evaluate model using lm-eval-harness (e.g. HellaSwag, ARC).
    """
    output_dir = f"{REPO_ID.split('/')[1].replace('-', '_')}_{tag.replace('.','_')}_{datetime.datetime.now(ZoneInfo('Africa/Cairo')).strftime('%Y_%m_%dT%H_%M_%S')}"
    tasks = [
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "winogrande",
        "piqa",
        "mmlu",
    ]

    cmd_parts = [
        "lm-eval",
        "--model hf",
        f"--model_args pretrained={REPO_ID},parallelize=True",
        f"--tasks {','.join(tasks)}",
        f"--device {device}",
        f"--batch_size 4",
        f"--output_path {output_dir}",
    ]

    # command = " ".join(cmd_parts) + " >/dev/null 2>&1"
    command = " ".join(cmd_parts)
    start_time = time.time()
    print(f"Running command: {command}")
    exit_code = os.system(command)
    score = 0
    # exit_code = 0
    # breakpoint()
    if exit_code == 0:
        log_score(tag, "lm_eval", score, output_dir)
    # breakpoint()
    benchmark_runtime = time.time() - start_time
    # breakpoint()
    return score


# === EVALUATION REGISTRY ===


def get_evaluator(task: str) -> Callable:
    if task == "fineweb":
        return evaluate_fineweb
    elif task in ["hellaswag", "arc_easy", "arc_challenge", "lm_eval"]:
        return evaluate_with_lm_harness
    else:
        raise ValueError(f"Unsupported evaluation task: {task}")


# === MAIN LOOP ===


def evaluate_all_tags_once():
    result = R2.list_objects_v2(Bucket=BUCKET, Prefix="", Delimiter="/")

    # Extract subfolders like epoch-0/, epoch-1/, etc.
    folders = [
        o.get("Prefix").rstrip("/").split("/")[-1]
        for o in result.get("CommonPrefixes", [])
        if o.get("Prefix").startswith("epoch-")
    ]

    # Sort by epoch number (epoch-0, epoch-1, ...)
    epochs = sorted(folders, key=lambda x: int(x.split("-")[1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in epochs:
        epoch = epoch.split("-")[1]
        tag = f"{__run__}.{epoch}.0"
        try:
            if tag.split(".")[0] != __run__:
                continue
            else:
                print(f"\n=== [TAG] {tag} ===")

            for task in EVAL_TYPES:
                if tag_exists(tag, task):
                    print(f"[✓] {task}: already evaluated")
                    continue

                if task != "fineweb":
                    continue

                print(f"[⏳] Evaluating {task}...")
                evaluator = get_evaluator(task)
                score = evaluator(device, tag)
                print(f"[✅] {task}: {score:.4f}")

        except Exception as e:
            print(f"[⚠️] Error evaluating tag {tag}: {e}")


# === Optional Continuous Mode ===


def monitor_repo(poll_interval_sec: int = 18000):
    print("[🔁] Starting continuous monitoring...")
    while True:
        evaluate_all_tags_once()
        print(f"[⏳] Sleeping for {poll_interval_sec}s...")
        time.sleep(poll_interval_sec)


# === Entry ===

if __name__ == "__main__":
    monitor_repo()
